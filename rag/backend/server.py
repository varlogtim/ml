import os
import yaml
import logging

from .enabler import Enabler
from pathlib import Path
from typing import Any
from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> dict[str, Any]:
        return self.config


class Server:
    def __init__(self, app: Enabler):
        # TODO Figure out parameterization
        self.path_to_frontend = "../frontend/dist/"
        self.flask = Flask(__name__)
        self._register_routes()
        self.app = app

    def _register_routes(self):
        self.flask.add_url_rule("/query", methods=["POST"], view_func=self.handle_query)
        self.flask.add_url_rule("/config", methods=["GET", "POST"], view_func=self.handle_config)
        self.flask.add_url_rule("/shell", methods=["POST"], view_func=self.handle_shell)
        # TODO, make a GET that tells you how to use this?
        # TODO make a GET/POST for config?

        self.flask.add_url_rule("/assets/<filename>", view_func=self.serve_assets)
        self.flask.add_url_rule("/", defaults={"path": ""}, view_func=self.serve_react_app)
        self.flask.add_url_rule("/<path:path>", view_func=self.serve_react_app)

    def handle_config(self):
        if request.method == "GET":
            return jsonify({"config": self.app.to_json()})
        # POST
        if not request.is_json:
            return jsonify({"error": "JSON payload required"}), 400
        data = request.get_json()
        config = data.get("config")  # Should be a string of json
        if not config:
            return jsonify({"error": "Must have 'config' key in JSON payload"}), 400
        try:
            new_app = Enabler.from_json(config)
            self.app = new_app
            logger.info(f"Applied new config: {self.app.to_json()}")
            return jsonify({"config": self.app.to_json()})
        except Exception as e:
            return jsonify({"error": f"Failed to apply config: {e}"}), 500
    
    def handle_query(self):
        if not request.is_json:
            return jsonify({"error": "JSON payload required"}), 400
        data = request.get_json()
        if not data.get("text"):
            return jsonify({"error": "Must have 'text' key in JSON payload"}), 400
        try:
            return jsonify({"response": self.app.query(data.get("text"))})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def handle_shell(self):
        if not request.is_json:
            return jsonify({"error": "JSON payload required"}), 400
        data = request.get_json()
        cmd = data.get("cmd")  # Should be a string of json
        if not cmd:
            return jsonify({"error": "Must have 'cmd' key in JSON payload"}), 400
        try:
            return jsonify({"cmd": cmd, "output": self.app.shell(cmd)})
        except Exception as e:
            return jsonify({"error": f"Failed to execute command: {e}"}), 500

    def serve_react_app(self, path: str = ""):
        # Serve index.html for SPA routes
        return send_from_directory(self.path_to_frontend, "index.html")

    def serve_assets(self, filename: str):
        # Serve files from dist/assets
        return send_from_directory(os.path.join(self.path_to_frontend, "assets"), filename)

