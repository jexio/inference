import argparse

import anyio

from inference.core.actor.schemas import Image
from inference.core.pipeline.face import FaceDetectionPipelineRunner
from inference.core.executor.triton import TritonExecutor
from inference.core.engine.triton import TritonEngine
from inference.settings import TritonEngineSettings, TritonExecutorSettings

import tritonclient.grpc.aio as grpcclient

parser = argparse.ArgumentParser()
parser.add_argument("-p",
                    "--path",
                    type=str,
                    required=True,
                    help="The absolute path to an image.")
flags = parser.parse_args()


def load_rgb(path):
    """Load image.

    Args:
        path: Path to an image.

    Returns:
        An image in rgb format.
    """
    import cv2
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


async def main():

    path = flags.path
    image = load_rgb(path)
    batch_size = 2
    settings = TritonEngineSettings(url="localhost:8001", verbose=False)
    connection = grpcclient.InferenceServerClient(
        settings.url,
        settings.verbose,
        settings.ssl,
        settings.root_certificates,
        settings.private_key,
        settings.certificate_chain,
    )
    engine = TritonEngine(connection)
    settings = TritonExecutorSettings()
    executor = TritonExecutor(engine, settings)
    runner = FaceDetectionPipelineRunner(executor)
    await runner.with_pipeline("face_detection_scrface")
    data = [Image(image.copy()) for _ in range(0, batch_size)]
    async for item in runner.run(data):
        print(item)
        print("#" * 100)

if __name__ == "__main__":
    anyio.run(main)
