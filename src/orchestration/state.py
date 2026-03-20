from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus
from src.models.task import ExperimentPlan


class ExperimentTree:
    """
    Graph-structured experiment lineage.
    Nodes are ExperimentNode objects linked by parent_id/children.
    Edge labels describe what changed between parent and child — Graph RAG compatible.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ExperimentNode] = {}

    def _new_id(self) -> str:
        return f"node_{uuid.uuid4().hex[:8]}"

    def add_root(self, plan: ExperimentPlan, stage: NodeStage = NodeStage.WARMUP) -> ExperimentNode:
        node = ExperimentNode(
            node_id=self._new_id(),
            parent_id=None,
            children=[],
            edge_label=None,
            stage=stage,
            status=NodeStatus.PENDING,
            plan=plan,
            depth=0,
            created_at=datetime.now(),
        )
        self._nodes[node.node_id] = node
        return node

    def add_child(
        self,
        parent_id: str,
        plan: ExperimentPlan,
        edge_label: str,
        stage: NodeStage = NodeStage.OPTIMIZE,
    ) -> ExperimentNode:
        parent = self._nodes[parent_id]
        node = ExperimentNode(
            node_id=self._new_id(),
            parent_id=parent_id,
            children=[],
            edge_label=edge_label,
            stage=stage,
            status=NodeStatus.PENDING,
            plan=plan,
            depth=parent.depth + 1,
            created_at=datetime.now(),
        )
        self._nodes[node.node_id] = node
        # Add child reference to parent (Pydantic model — must reassign)
        updated_parent = parent.model_copy(update={"children": parent.children + [node.node_id]})
        self._nodes[parent_id] = updated_parent
        return node

    def get_node(self, node_id: str) -> ExperimentNode:
        return self._nodes[node_id]

    def get_roots(self) -> List[ExperimentNode]:
        return [n for n in self._nodes.values() if n.parent_id is None]

    def get_leaves(self) -> List[ExperimentNode]:
        return [n for n in self._nodes.values() if not n.children]

    def get_path_to_root(self, node_id: str) -> List[ExperimentNode]:
        path = []
        current_id: Optional[str] = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def get_incumbent(self, higher_is_better: bool = True) -> Optional[ExperimentNode]:
        candidates = [
            n for n in self._nodes.values()
            if n.has_result()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: (
            (n.primary_metric() or 0) if higher_is_better else -(n.primary_metric() or 0)
        ))

    def update_node(self, node: ExperimentNode) -> None:
        """Replace a node (e.g. after attaching ExperimentRun or changing status)."""
        self._nodes[node.node_id] = node

    def all_nodes(self) -> List[ExperimentNode]:
        return list(self._nodes.values())

    def save(self, path: Union[str, Path]) -> None:
        """Serialize tree to JSON for persistence and future Graph RAG indexing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: json.loads(n.model_dump_json()) for nid, n in self._nodes.items()}
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentTree":
        data = json.loads(Path(path).read_text())
        tree = cls()
        for nid, node_data in data["nodes"].items():
            tree._nodes[nid] = ExperimentNode.model_validate(node_data)
        return tree
