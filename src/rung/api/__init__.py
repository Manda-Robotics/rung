"""Client side of Rung as a hosted robot-policy API.

The server hosts the same harness that runs locally: a robot (or a RoboLab evaluator)
sends observations over a WebSocket and receives the next action. The harness, the model
key and the model conversation stay on the server; the robot keeps grading and safety.
"""

PROTOCOL_VERSION = "rung-api/0.1"
