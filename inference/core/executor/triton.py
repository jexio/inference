from typing import List, Tuple

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from inference.core.engine.interface import ModelConfig, ModelMetadata
from inference.core.engine.triton import TritonEngine
from inference.core.exceptions import (TritonInferenceServerException,
                                       TritonUnableToGetModelException,
                                       TritonInferenceClientException)
from inference.core.executor import Executor
from inference.core.schemas import TritonExecutorInput
from inference.core.utils.decorator import inference_executor_exception_handler
from inference.core.type_hints import ActorResponse, BatchedNDArray
from inference.settings import TritonExecutorSettings


class TritonExecutor(Executor):
    """An Executor is used to perform any kind of communication with the InferenceServer using gRPC protocol."""

    def __init__(self, engine: TritonEngine, settings: TritonExecutorSettings) -> None:
        """Create a new instance of TritonExecutor.

        Args:
            engine: An instance of :class:`inference.core.engine.Engine` to communicate with inference server using gRPC protocol.
            settings: An instance of :class:`inference.core.schemas.TritonExecutorSettings` class.
        """
        self._engine = engine
        self._settings = settings
        self._initialize()

    @staticmethod
    def _prepare_data(
        data: Tuple[BatchedNDArray],
        inputs: Tuple[TritonExecutorInput],
        outputs: Tuple[str],
    ) -> Tuple[List[grpcclient.InferInput], List[grpcclient.InferRequestedOutput]]:
        """Prepare inputs, outputs.

        Args:
            data: The tensor data in numpy array format.
            inputs: An object of TritonExecutorInput class is used to describe input tensor for an inference request.
            outputs: Names of output tensor to associate with this object.

        Raises:
            TritonInferenceClientException: The number of inputs is not equal to the number of elements in the `data`

        Returns:
            A list of InferInput objects, each describing data for a input tensor required by the model.
            A list of InferRequestedOutput objects, each describing how the output data must be returned.
        """
        grpc_inputs: List[grpcclient.InferInput] = list()
        if len(data) != len(inputs):
            raise TritonInferenceClientException(
                "The number of inputs is not equal to the number of elements in the `data` to be processed."
            )

        for items, input_ in zip(data, inputs):
            element = np.array(items)
            element = element.astype(input_.datatype.value)
            grpc_input = grpcclient.InferInput(input_.name, element.shape, np_to_triton_dtype(element.dtype))
            grpc_input.set_data_from_numpy(element)
            grpc_inputs.append(grpc_input)

        grpc_outputs = [grpcclient.InferRequestedOutput(output) for output in outputs]
        return grpc_inputs, grpc_outputs

    @staticmethod
    def _postprocess_infer(outputs: Tuple[str], result: grpcclient.InferResult) -> ActorResponse:
        """Postprocess the result of the inference.

        Args:
            outputs: Names of output tensor to associate with this object.
            result: The object holding the result of the inference.

        Returns:
            InferResult converted to ndarray.
        """
        inference_outputs: ActorResponse = dict()
        for output in outputs:
            data = result.as_numpy(output)
            inference_outputs[output] = data

        return inference_outputs

    def _initialize(self) -> None:
        """Initialize specific behavior here."""
        # To make sure no shared memory regions are registered with the server.
        if self._settings.use_shared_memory:
            self._engine.unregister_system_shared_memory()
            self._engine.unregister_cuda_shared_memory()

    async def _check_server(self, model_name: str, model_version: str) -> None:
        """Check the availability of the server and the model.

        Args:
            model_name: Model to health check.
            model_version: Model version to health check.

        Raises:
             TritonInferenceServerException: if something went wrong.
                (no connectivity, server or model is not ready to work).
            TritonUnableToGetModelException: if model does not exist or model is unloaded.
        """
        try:
            if not await self._engine.is_available():
                raise TritonInferenceServerException("Triton inference server is not alive.")

        except InferenceServerException as e:  # noqa: B902
            raise TritonInferenceServerException(f"non-OK-status RPC termination: {str(e)}")

        try:
            _ = await self.get_model_metadata(model_name, model_version)
        except InferenceServerException as e:
            is_incorrect_name = "Request for unknown model" in str(e)
            if is_incorrect_name:
                raise TritonInferenceServerException(
                    f"The model - {model_name} "
                    f"with version - {model_version} does not exist on the current server."
                )
            raise TritonInferenceServerException(str(e))

        try:
            if not await self._engine.is_model_ready(model_name, model_version):
                raise TritonUnableToGetModelException(
                    f"Unable to load model. " f"Model - {model_name}, version - {model_version}."
                )
        except InferenceServerException as e:  # noqa: B902
            raise TritonInferenceServerException(f"non-OK-status RPC termination: {str(e)}")

    async def run(
        self,
        model_name: str,
        model_version: str,
        data: Tuple[BatchedNDArray],
        inputs: Tuple[TritonExecutorInput],
        outputs: Tuple[str],
    ) -> ActorResponse:
        """Send batch of events as a requests to Triton inference server using triton client API.

        Args:
            model_name: Name of the model specifies which model can handle the inference requests that are sent to Triton inference server.
            model_version: Version of the model.
            data: The tensor data in numpy array format.
            inputs: An object of TritonExecutorInput class is used to describe input tensor for an inference request.
            outputs: Names of output tensor to associate with this object.

        Returns:
            A result of the inference.
        """
        await self._check_server(model_name, model_version)
        grpc_inputs, grpc_outputs = self._prepare_data(data, inputs, outputs)
        result = await self._engine.run(model_name, model_version, grpc_inputs, grpc_outputs, self._settings.timeout)
        result = self._postprocess_infer(outputs, result)
        return result

    @inference_executor_exception_handler
    async def get_model_metadata(self, model_name: str, model_version: str) -> ModelMetadata:
        """Contact the inference server and get the metadata for specified model."""
        return await self._engine.get_model_metadata(model_name, model_version)

    @inference_executor_exception_handler
    async def get_model_config(self, model_name: str, model_version: str) -> ModelConfig:
        """Contact the inference server and get the configuration for specified model."""
        return await self._engine.get_model_config(model_name, model_version)


__all__ = ("TritonExecutor",)
