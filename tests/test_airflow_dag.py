"""
Test script to verify that the Airflow DAG can be loaded and parsed without errors.
"""
import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Airflow modules
try:
    from airflow.models import DagBag
except ImportError:
    print("Airflow is not installed. Skipping Airflow DAG tests.")
    sys.exit(0)

class TestAirflowDag(unittest.TestCase):
    """Test class for Airflow DAGs."""

    def setUp(self):
        """Set up the test environment."""
        self.dagbag = DagBag(dag_folder=str(Path(__file__).parent.parent / 'dags'), include_examples=False)

    def test_dag_loaded(self):
        """Test that the DAG is loaded without errors."""
        self.assertFalse(
            len(self.dagbag.import_errors),
            f"DAG import errors: {self.dagbag.import_errors}"
        )
        
    def test_cf_ingest_dag_loaded(self):
        """Test that the cf_ingest DAG is loaded."""
        self.assertIn(
            'cf_ingest',
            self.dagbag.dag_ids,
            "cf_ingest DAG not found in the DAG bag"
        )
        
    def test_cf_ingest_dag_tasks(self):
        """Test that the cf_ingest DAG has the expected tasks."""
        dag = self.dagbag.get_dag('cf_ingest')
        self.assertIsNotNone(dag, "cf_ingest DAG not found")
        
        expected_tasks = [
            'fetch_new_scrobbles',
            'enrich_artist_info',
            'clean_artist_data',
            'run_canonization',
            'export_to_parquet',
            'run_data_profiling'
        ]
        
        for task_id in expected_tasks:
            self.assertIn(
                task_id,
                [task.task_id for task in dag.tasks],
                f"Task {task_id} not found in cf_ingest DAG"
            )
            
    def test_cf_ingest_dag_dependencies(self):
        """Test that the cf_ingest DAG has the expected dependencies."""
        dag = self.dagbag.get_dag('cf_ingest')
        self.assertIsNotNone(dag, "cf_ingest DAG not found")
        
        # Define the expected task dependencies
        expected_dependencies = {
            'fetch_new_scrobbles': ['enrich_artist_info'],
            'enrich_artist_info': ['clean_artist_data'],
            'clean_artist_data': ['run_canonization'],
            'run_canonization': ['export_to_parquet'],
            'export_to_parquet': ['run_data_profiling'],
        }
        
        # Check that each task has the expected downstream tasks
        for task_id, downstream_task_ids in expected_dependencies.items():
            task = dag.get_task(task_id)
            downstream_tasks = task.downstream_list
            downstream_task_ids_actual = [t.task_id for t in downstream_tasks]
            
            for expected_downstream_task_id in downstream_task_ids:
                self.assertIn(
                    expected_downstream_task_id,
                    downstream_task_ids_actual,
                    f"Task {task_id} should have {expected_downstream_task_id} as a downstream task"
                )

if __name__ == '__main__':
    unittest.main()