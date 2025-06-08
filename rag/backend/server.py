import os
import yaml
import logging

from .enabler import Enabler
from pathlib import Path
from typing import Any
from flask import Flask, jsonify, request

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
        self.flask = Flask(__name__)
        self._register_routes()
        self.app = app

    def _register_routes(self):
        self.flask.add_url_rule("/query", methods=["POST"], view_func=self.query)
        # TODO, make a GET that tells you how to use this?
        # TODO make a GET/POST for config?
    
    def query(self):
        if not request.is_json:
            return jsonify({"error": "JSON payload required"}), 400
        data = request.get_json()
        if not data.get("text"):
            return jsonify({"error": "Must have 'text' key in JSON payload"}), 400
        try:
            return jsonify({"response": self.app.query(data.get("text"))})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


