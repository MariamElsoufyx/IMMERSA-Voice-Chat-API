from typing import Dict
from fastapi import WebSocket

from app.services.streaming.stream_session import StreamSession


class ConnectionManager:
    def __init__(self):
        # session_id -> websocket
        self.active_connections: Dict[str, WebSocket] = {}

        # session_id -> StreamSession
        self.sessions: Dict[str, StreamSession] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]

        print(f"🔴 [DISCONNECTED] sid={session_id[:8]} | session closed")

    def create_session(self, session_id: str) -> StreamSession:
        session = StreamSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamSession:
        return self.sessions.get(session_id)

    async def send_json(self, session_id: str, data: dict):
        websocket = self.active_connections.get(session_id)

        if websocket:
            await websocket.send_json(data)
        else:
            print(f"⚠️ MANAGER:  [SEND FAILED] No websocket for sid={session_id[:8]}")

    async def broadcast(self, data: dict):
        for session_id, websocket in self.active_connections.items():
            await websocket.send_json(data)
            print(f"📡 [BROADCAST] sid={session_id[:8]} | data_type={data.get('type')}")
