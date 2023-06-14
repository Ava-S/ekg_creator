from ..cypher_queries.query_translators import ClassCypher, ConditionCypher, EntityCypher, \
    RelationConstructorByNodesCypher, RelationCypher, RelationConstructorByRelationsCypher, \
    RelationConstructorByQueryCypher, EntityConstructorByQueryCypher, EntityConstructorByRelationCypher, \
    EntityConstructorByNodesCypher, RelationshipCypher, LogCypher


class Interpreter:
    def __init__(self, query_language):
        self.class_qi = None
        self.condition_qi = None
        self.entity_qi = None
        self.relation_qi = None
        self.relation_constructor_by_nodes_qi = None
        self.relation_constructor_by_relations_qi = None
        self.relation_constructor_by_query_qi = None
        self.entity_constructor_by_query_qi = None
        self.entity_constructor_by_relation_qi = None
        self.entity_constructor_by_nodes_qi = None
        self.relationship_qi = None
        self.nodes_qi = None
        self.log_qi = None

        self.set_interpreters(query_language)

    def set_interpreters(self, query_language):
        if query_language == "Cypher":
            self.class_qi = ClassCypher
            self.condition_qi = ConditionCypher
            self.entity_qi = EntityCypher
            self.relation_qi = RelationCypher
            self.relation_constructor_by_nodes_qi = RelationConstructorByNodesCypher
            self.relation_constructor_by_relations_qi = RelationConstructorByRelationsCypher
            self.relation_constructor_by_query_qi = RelationConstructorByQueryCypher
            self.entity_constructor_by_query_qi = EntityConstructorByQueryCypher
            self.entity_constructor_by_relation_qi = EntityConstructorByRelationCypher
            self.entity_constructor_by_nodes_qi = EntityConstructorByNodesCypher
            self.relationship_qi = RelationshipCypher
            self.log_qi = LogCypher
