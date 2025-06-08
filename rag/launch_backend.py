import logging

from backend.server import ConfigParser, Server
from backend.enabler import Enabler


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# TODO there is still that matter of the path to the config file.
if __name__ == "__main__":
    config = ConfigParser("default.yaml")
    app = Enabler(config)
    logger.info(f"Loaded App: '{app.app_name}' with config: '{app.to_json()}'")

    server = Server(app)
    logger.info(f"Loaded Flask Server")
    server.flask.run(debug=True, host="0.0.0.0", port=5000)
