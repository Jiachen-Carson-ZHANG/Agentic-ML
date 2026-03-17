import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.orchestration.state import ExperimentTree
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus
from src.models.task import ExperimentPlan


def make_plan(metric="roc_auc", rationale="test") -> ExperimentPlan:
    return ExperimentPlan(
        eval_metric=metric, model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale=rationale
    )


def test_add_root_node():
    tree = ExperimentTree()
    node = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    assert node.parent_id is None
    assert node.depth == 0
    assert node.edge_label is None
    assert tree.get_node(node.node_id) is node


def test_add_child_node():
    tree = ExperimentTree()
    root = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    child = tree.add_child(
        parent_id=root.node_id,
        plan=make_plan(metric="f1_macro"),
        edge_label="changed eval_metric to f1_macro due to class imbalance",
        stage=NodeStage.OPTIMIZE,
    )
    assert child.parent_id == root.node_id
    assert child.depth == 1
    assert "f1_macro" in child.edge_label
    assert child.node_id in tree.get_node(root.node_id).children


def test_get_roots():
    tree = ExperimentTree()
    r1 = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    r2 = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    roots = tree.get_roots()
    assert len(roots) == 2
    assert r1 in roots and r2 in roots


def test_get_path_to_root():
    tree = ExperimentTree()
    root = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    child = tree.add_child(root.node_id, make_plan(), "changed metric", NodeStage.OPTIMIZE)
    grandchild = tree.add_child(child.node_id, make_plan(), "changed features", NodeStage.OPTIMIZE)
    path = tree.get_path_to_root(grandchild.node_id)
    assert path[0].node_id == root.node_id
    assert path[-1].node_id == grandchild.node_id


def test_get_incumbent_none_when_empty():
    tree = ExperimentTree()
    tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    assert tree.get_incumbent(higher_is_better=True) is None


def test_serialize_preserves_edge_labels(tmp_path):
    tree = ExperimentTree()
    root = tree.add_root(plan=make_plan(), stage=NodeStage.WARMUP)
    tree.add_child(root.node_id, make_plan(), "changed eval_metric", NodeStage.OPTIMIZE)
    path = tmp_path / "tree.json"
    tree.save(path)
    data = json.loads(path.read_text())
    # Find the child node in serialized form
    nodes = data["nodes"]
    child_nodes = [n for n in nodes.values() if n.get("edge_label")]
    assert len(child_nodes) == 1
    assert child_nodes[0]["edge_label"] == "changed eval_metric"
