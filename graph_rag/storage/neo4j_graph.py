from collections import defaultdict

import networkx as nx
from neo4j import GraphDatabase

# Les labels et types de relations ne peuvent pas etre parametres en Cypher :
# seules ces valeurs connues sont injectees dans les requetes.
NODE_LABELS = ("Class", "Function", "External")
RELATIONS = ("CALLS", "INHERITS_FROM", "HAS_METHOD")
BATCH_SIZE = 1000

SCHEMA_QUERIES = (
    "CREATE CONSTRAINT code_node_uid IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT code_project_id IF NOT EXISTS FOR (p:CodeProject) REQUIRE p.project_id IS UNIQUE",
    "CREATE INDEX code_node_project IF NOT EXISTS FOR (n:CodeNode) ON (n.project_id)",
)


def _uid(project_id, node_id):
    """Identifiant unique d'un noeud tous projets confondus."""
    return f"{project_id}:{node_id}"


def _batches(rows):
    for i in range(0, len(rows), BATCH_SIZE):
        yield rows[i:i + BATCH_SIZE]


class Neo4jGraphStore:
    """Graphe du code dans Neo4j.

    Modele : (:CodeNode:Class|Function|External {uid, project_id, node_id, type, file, doc, source})
    relies par [:CALLS], [:INHERITS_FROM], [:HAS_METHOD] ; (:CodeProject {project_id}) marque
    un projet completement ecrit.
    """

    name = "Neo4j"

    def __init__(self, uri, user, password, database="neo4j"):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._schema_ready = False

    def close(self):
        self._driver.close()

    def _query(self, query, **params):
        records, _, _ = self._driver.execute_query(query, params, database_=self._database)
        return records

    def _ensure_schema(self):
        if not self._schema_ready:
            for query in SCHEMA_QUERIES:
                self._query(query)
            self._schema_ready = True

    def has_project(self, project_id):
        self._ensure_schema()
        records = self._query(
            "MATCH (p:CodeProject {project_id: $project_id}) RETURN count(p) > 0 AS found",
            project_id=project_id,
        )
        return records[0]["found"]

    def write_graph(self, project_id, graph):
        """Ecrit noeuds, relations puis le marqueur CodeProject dans une seule transaction."""
        self._ensure_schema()

        nodes_by_label = defaultdict(list)
        for node, data in graph.nodes(data=True):
            label = data.get("type") if data.get("type") in NODE_LABELS else "External"
            nodes_by_label[label].append({
                "uid": _uid(project_id, node),
                "node_id": str(node),
                "file": data.get("file"),
                "doc": data.get("doc"),
                "source": data.get("source"),
            })

        edges_by_relation = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            relation = data.get("relation", "CALLS")
            if relation in RELATIONS:
                edges_by_relation[relation].append({"src": _uid(project_id, u), "dst": _uid(project_id, v)})

        def write(tx):
            for label, rows in nodes_by_label.items():
                for batch in _batches(rows):
                    tx.run(
                        "UNWIND $rows AS row "
                        "MERGE (n:CodeNode {uid: row.uid}) "
                        f"SET n += row, n.project_id = $project_id, n.type = $label, n:{label}",
                        rows=batch, project_id=project_id, label=label,
                    )
            for relation, rows in edges_by_relation.items():
                for batch in _batches(rows):
                    tx.run(
                        "UNWIND $rows AS row "
                        "MATCH (a:CodeNode {uid: row.src}) "
                        "MATCH (b:CodeNode {uid: row.dst}) "
                        f"MERGE (a)-[:{relation}]->(b)",
                        rows=batch,
                    )
            tx.run(
                "MERGE (p:CodeProject {project_id: $project_id}) "
                "SET p.node_count = $node_count, p.edge_count = $edge_count, p.indexed_at = datetime()",
                project_id=project_id,
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
            )

        with self._driver.session(database=self._database) as session:
            session.execute_write(write)

    def expand(self, project_id, seeds, max_depth=1):
        subgraph = nx.DiGraph()
        if not seeds:
            return subgraph

        relation_pattern = "|".join(RELATIONS)
        nodes = self._query(
            "UNWIND $uids AS uid "
            "MATCH (s:CodeNode {uid: uid}) "
            f"MATCH path = (s)-[:{relation_pattern}*0..{int(max_depth)}]-(n:CodeNode) "
            "WITH n, min(length(path)) AS depth "
            "RETURN n.uid AS uid, n.node_id AS node_id, n.type AS type, n.file AS file, "
            "n.doc AS doc, n.source AS source, depth",
            uids=[_uid(project_id, seed) for seed in seeds],
        )
        uids = [record["uid"] for record in nodes]
        edges = self._query(
            "MATCH (a:CodeNode)-[r]->(b:CodeNode) "
            "WHERE a.uid IN $uids AND b.uid IN $uids "
            "RETURN a.node_id AS src, b.node_id AS dst, type(r) AS relation",
            uids=uids,
        )

        for record in nodes:
            subgraph.add_node(record["node_id"], type=record["type"], file=record["file"],
                              doc=record["doc"], source=record["source"], depth=record["depth"])
        for record in edges:
            subgraph.add_edge(record["src"], record["dst"], relation=record["relation"])
        return subgraph
