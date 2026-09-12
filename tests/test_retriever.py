"""
test_retriever.py
-----------------
Unit tests for session persistence, markdown exports, and document chunking.
"""

import os
import shutil
import unittest
from retriever import (
    save_chat_session,
    delete_chat_session,
    load_all_chat_sessions,
    export_session_to_markdown,
    prepare_chat_documents,
)


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_past_chats"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_session_lifecycle(self):
        prompts = ["What is quantum computing?", "How do qubits work?"]
        replies = ["Quantum computing uses quantum mechanics...", "Qubits exist in superposition..."]

        # 1. Save session
        sid = save_chat_session(
            prompts=prompts,
            replies=replies,
            folder_path=self.test_dir,
            session_id="test_session_001",
            latest_contexts=["### Wikipedia\nQubit definition"],
        )
        self.assertEqual(sid, "test_session_001")

        # 2. Load all sessions
        sessions = load_all_chat_sessions(folder_path=self.test_dir)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "test_session_001")
        self.assertEqual(len(sessions[0]["chats"]), 2)

        # 3. Document preparation
        docs = prepare_chat_documents(sessions=sessions)
        self.assertGreaterEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["session_id"], "test_session_001")

        # 4. Markdown export
        md = export_session_to_markdown(prompts, replies, session_id=sid)
        self.assertIn("# JANE Research Dossier", md)
        self.assertIn("What is quantum computing?", md)
        self.assertIn("How do qubits work?", md)

        # 5. Delete session
        deleted = delete_chat_session(sid, folder_path=self.test_dir)
        self.assertTrue(deleted)
        self.assertEqual(len(load_all_chat_sessions(folder_path=self.test_dir)), 0)


if __name__ == "__main__":
    unittest.main()
