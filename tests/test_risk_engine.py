import unittest
import sqlite3
from pathlib import Path
import main
from core.risk_engine import predict_risk
from database.db_manager import DBManager


class TestRiskEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = Path(__file__).parent / "test_isg_audit.db"
        cls.test_db = DBManager(db_path=cls.test_db_path)
        cls.original_db = main.db

    @classmethod
    def tearDownClass(cls):
        main.db = cls.original_db
        try:
            if cls.test_db_path.exists():
                cls.test_db_path.unlink()
        except Exception:
            pass

    def setUp(self):
        main.db = self.test_db
        self.db = self.test_db
        # Her test öncesi test tablosundaki kayıtları temizle
        with sqlite3.connect(self.db.db_path) as conn:
            conn.cursor().execute("DELETE FROM audit_logs")
            conn.commit()

    def test_predict_risk_returns_valid_label(self):
        result = predict_risk(1, 1, 1, 0)
        self.assertIsInstance(result, str)
        self.assertIn(result, ["High", "Medium", "Low"])

    def test_high_risk_scenario_returns_high(self):
        result = predict_risk(5, 5, 3, 1)
        self.assertEqual(result, "High")

    def test_low_risk_scenario_returns_low(self):
        result = predict_risk(0, 0, 0, 0)
        self.assertEqual(result, "Low")

    def test_add_log_returns_id_and_fetches_record(self):
        log_id = self.db.add_log("Test Tesisi", 1, 2, 5.5, "Medium")
        logs = self.db.fetch_logs()
        self.assertIsNotNone(log_id)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0][0], log_id)

    def test_run_audit_adds_record_to_database(self):
        log_id, risk_level = main.run_audit("Uctan Uca Test", 5, 5, 3, 1)
        logs = self.db.fetch_logs()
        self.assertIsNotNone(log_id)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0][0], log_id)


if __name__ == "__main__":
    unittest.main()
