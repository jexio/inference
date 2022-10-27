from typing import List, Optional, Tuple

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

from inference.core.engine.interface import Engine, ModelConfig, ModelMetadata

from inference.core.engine.schemas import EngineHealthStatus, EngineHealthDetails
from inference.core.exceptions import TritonInferenceServerException
from inference.core.interface import HealthStatus, Status, TModelStatistic


class TritonEngine(Engine[Tuple[str]]):
    """Engine communicating with inference serving software using gRPC protocol."""

    def __init__(self, connection: grpcclient.InferenceServerClient) -> None:
        """Create a new instance of TritonEngine.

        Args:
            connection: An object is used to perform any kind of communication with the triton inference server using gRPC protocol.
        """
        self._connection = connection

    def __str__(self) -> str:
        """Returns string representation of TritonEngine."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns object representation of TritonEngine."""
        return "TritonEngine"

    async def is_available(self) -> bool:
        """Contact the inference server and get liveness."""
        return await self._connection.is_server_live()

    async def is_ready(self) -> bool:
        """Contact the inference server and get readiness."""
        return await self._connection.is_server_ready()

    async def is_model_ready(self, name: str, version: str) -> bool:
        """Contact the inference server and get the readiness of specified model.

        Args:
            name: Name of the model specifies which model can handle the inference requests that are sent to Triton inference server.
            version: Version of the model.

        Returns:
            True if the model is ready, False if not ready.
        """
        return await self._connection.is_model_ready(name, version)

    async def is_supported(self) -> bool:
        """Check if server is available."""
        try:
            return await self.is_ready()
        except TritonInferenceServerException:
            return False

    def get_name(self) -> str:
        """Get the name of the engine."""
        return self.__str__()

    async def get_health(self) -> HealthStatus:
        """Get the health status of the engine.

        Returns:
            UP if alive DOWN otherwise.
        """
        try:
            metadata = await self._connection.get_server_metadata(as_json=True)
            return EngineHealthStatus(
                status=Status.UP,
                details=EngineHealthDetails(
                    version=metadata["version"],
                ),
            )
        except InferenceServerException as e:
            return EngineHealthStatus(status=Status.DOWN, details=EngineHealthDetails(failure=str(e)))

    async def get_inference_statistics(
        self, model_name: Optional[str] = "", model_version: Optional[str] = ""
    ) -> TModelStatistic:
        """Get the inference statistics for the specified model name and version.

        Args:
            model_name: The name of the model to get statistics.
            model_version: The version of the model to get inference statistics.

        Returns:
            Inference statistics as a json dict

        Raises:
            TritonInferenceServerException: if something went wrong. (no connectivity, server or model is not ready to work).
        """
        try:
            return await self._connection.get_inference_statistics(model_name, model_version, as_json=True)
        except InferenceServerException as e:  # noqa: B902
            raise TritonInferenceServerException(f"non-OK-status RPC termination: {str(e)}")

    async def close(self) -> None:
        """Close the client. Any future calls to server will result in an error."""
        await self._connection.close()

    async def get_model_metadata(self, name: str, version: str) -> ModelMetadata:
        """Contact the inference server and get the metadata for specified model."""
        return await self._connection.get_model_metadata(name, version, as_json=True)

    async def get_model_config(self, name: str, version: str) -> ModelConfig:
        """Contact the inference server and get the configuration for specified model."""
        data = await self._connection.get_model_config(name, version, as_json=True)
        return data["config"]

    async def run(
        self,
        name: str,
        version: str,
        grpc_inputs: List[grpcclient.InferInput],
        grpc_outputs: List[grpcclient.InferRequestedOutput],
        timeout: Optional[float] = None,
    ) -> grpcclient.InferResult:
        """Send batch of events as a requests to Triton inference server using triton client API.

        Args:
            name: Name of the model specifies which model can handle the inference requests that are sent to Triton inference server.
            version: Version of the model.
            grpc_inputs: A list of InferInput objects, each describing data for a input tensor required by the model.
            grpc_outputs: A list of InferRequestedOutput objects, each describing how the output data must be returned.
            timeout: The maximum end-to-end time, in seconds, the request is allowed to take.

        Returns:
            A result of the inference.
        """
        result = await self._connection.infer(
            model_name=name,
            model_version=version,
            inputs=grpc_inputs,
            outputs=grpc_outputs,
            client_timeout=timeout,
        )
        return result


__all__ = (
    "TritonEngine",
)
