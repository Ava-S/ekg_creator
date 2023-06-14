from typing import Optional, List

from ..data_managers.semantic_header import Entity, Relation, Relationship, SemanticHeader
from ..database_managers.db_connection import DatabaseConnection
from ..utilities.performance_handling import Performance
from ..cypher_queries.query_library import CypherQueryLibrary


class EKGUsingSemanticHeaderBuilder:
    def __init__(self, db_connection: DatabaseConnection, semantic_header: SemanticHeader, batch_size: int,
                 perf: Performance):
        self.connection = db_connection
        self.semantic_header = semantic_header
        self.batch_size = batch_size
        self.perf = perf

    def _write_message_to_performance(self, message: str):
        if self.perf is not None:
            self.perf.finished_step(activity=message)

    def create_log(self):
        if self.semantic_header.log.include:
            self.connection.exec_query(CypherQueryLibrary.get_create_log_query)
            self._write_message_to_performance(message="Creation of (:Log) nodes")

            if self.semantic_header.log.has:
                self.connection.exec_query(CypherQueryLibrary.get_link_events_to_log_query,
                                           **{"batch_size": self.batch_size})
                self._write_message_to_performance(message="Creation of (:Event) <- [:HAS] - (:Log) relation")

    def create_entities(self, entity_types: Optional[List[str]]) -> None:
        if entity_types is None:
            entity_types = [entity.constructed_by.node_label for entity in
                            self.semantic_header.get_entities_constructed_by_nodes()]
        entity: Entity
        for entity in self.semantic_header.get_entities_constructed_by_nodes():
            if entity.constructed_by.node_label in entity_types:
                self.connection.exec_query(CypherQueryLibrary.get_create_entity_query, **{"entity": entity})
                self._write_message_to_performance(f"Entity (:{entity.get_label_string()}) node created")

    def create_nodes(self, node_names: Optional[List[str]]) -> None:
        if node_names is None:
            nodes = {name: node for name, node in self.semantic_header.constructors.items() if "Node" in name}
        else:
            nodes = {name: node for name, node in self.semantic_header.constructors.items() if name in node_names}

        for name, node in nodes.items():
            self.connection.exec_query(CypherQueryLibrary.get_create_constructor_query, **{"constructor": node})
            self._write_message_to_performance(f"Node (:{name}) created")

    def create_relationships(self, relationship_names: Optional[List[str]]) -> None:
        if relationship_names is None:
            relationships = {name: relationship for name, relationship in self.semantic_header.constructors.items() if
                     "Relation" in name}
        else:
            relationships = {name: node for name, node in self.semantic_header.constructors.items() if
                     name in relationship_names}

        for name, relationship in relationships.items():
            self.connection.exec_query(CypherQueryLibrary.get_create_constructor_query, **{"constructor": relationship})
            self._write_message_to_performance(f"Relationship [:{name}] created")

    def correlate_events_to_entities(self, entity_types: Optional[List[str]]) -> None:
        if entity_types is None:
            entity_types = [entity.constructed_by.node_label for entity in
                            self.semantic_header.get_entities_constructed_by_nodes()]

        # correlate events that contain a reference from an entity to that entity node
        entity: Entity
        for entity in self.semantic_header.get_entities_constructed_by_nodes():
            if entity.corr and entity.constructed_by.node_label in entity_types:
                # find events that contain the entity as property and not nan
                # save the value of the entity property as id and also whether it is a virtual entity
                # create a new entity node if it not exists yet with properties
                self.connection.exec_query(CypherQueryLibrary.get_correlate_events_to_entity_query,
                                           **{"entity": entity, "batch_size": self.batch_size})

                self._write_message_to_performance(
                    f"Relation (:Event) - [:CORR] -> (:{entity.get_label_string()}) created")

    def create_entity_relations_using_nodes(self, relation_types: Optional[List[str]]) -> None:
        # find events that are related to different entities of which one event also has a reference to the other entity
        # create a relation between these two entities
        if relation_types is None:
            relation_types = [relation.type for relation in self.semantic_header.get_relations_derived_from_nodes()]

        relation: Relation
        for relation in self.semantic_header.get_relations_derived_from_nodes():
            if relation.include and relation.type in relation_types:
                self.create_foreign_nodes(relation)
                self.create_relations_using_nodes(relation)
                self.delete_foreign_nodes(relation)

                self._write_message_to_performance(
                    message=f"Relation (:{relation.constructed_by.from_node_label}) - [:{relation.type}] -> "
                            f"(:{relation.constructed_by.to_node_label}) done")

    def create_foreign_nodes(self, relation: Relation):
        self.connection.exec_query(CypherQueryLibrary.create_foreign_key_relation,
                                   **{"relation": relation})
        self.connection.exec_query(CypherQueryLibrary.merge_foreign_key_nodes,
                                   **{"relation": relation})

    def create_relations_using_nodes(self, relation: Relation):
        self.connection.exec_query(CypherQueryLibrary.get_create_entity_relationships_query,
                                   **{
                                       "relation": relation,
                                       "batch_size": self.batch_size
                                   })

    def delete_foreign_nodes(self, relation: Relation):
        self.connection.exec_query(CypherQueryLibrary.get_delete_foreign_nodes_query,
                                   **{"relation": relation})

    def create_entity_relations_using_relations(self, relation_types: Optional[List[str]]) -> None:
        if relation_types is None:
            relation_types = [relation.type for relation in self.semantic_header.get_relations_derived_from_relations()]
        relation: Relation
        for relation in self.semantic_header.get_relations_derived_from_relations():
            if relation.include and relation.type in relation_types:
                self.connection.exec_query(CypherQueryLibrary.get_create_relation_by_relations_query,
                                           **{
                                               "relation": relation,
                                               "batch_size": self.batch_size
                                           })

    def create_entities_by_relations(self, entity_types: Optional[List[str]]) -> None:
        relation: Relation
        entity: Entity
        if entity_types is None:
            entity_types = [entity.type for entity in self.semantic_header.get_entities_constructed_by_relations()]
        for entity in self.semantic_header.get_entities_constructed_by_relations():
            if entity.include and entity.type in entity_types:
                self.connection.exec_query(CypherQueryLibrary.get_create_entities_by_relations_query,
                                           **{"entity": entity})

                self.connection.exec_query(CypherQueryLibrary.get_add_reified_relation_query,
                                           **{"entity": entity, "batch_size": self.batch_size})
                self._write_message_to_performance(
                    message=f"Relation [:{entity.type.upper()}] reified as "
                            f"(:Entity:{entity.get_label_string()}) node")

    def correlate_events_to_reification(self) -> None:
        reified_entity: Entity
        for reified_entity in self.semantic_header.get_entities_constructed_by_relations():
            if reified_entity.corr:
                reified_entity_labels = reified_entity.get_label_string()
                # correlate events that are related to an entity which is reified into a new entity
                # to the new reified entity

                self.connection.exec_query(CypherQueryLibrary.get_correlate_events_to_reification_query,
                                           **{"reified_entity": reified_entity})

                self._write_message_to_performance(
                    f"Relation (:Event) - [:CORR] -> (:Entity:{reified_entity_labels}) created")

    def create_df_edges(self, entity_types) -> None:
        entity: Entity

        if entity_types is None:
            entity_types = [entity.type for entity in self.semantic_header.entities]

        for entity in self.semantic_header.entities:
            if entity.df and entity.type in entity_types:
                self.connection.exec_query(CypherQueryLibrary.get_create_directly_follows_query,
                                           **{"entity": entity, "batch_size": self.batch_size})
                self._write_message_to_performance(f"Created [:DF] edge for (:{entity.get_label_string()})")

    def merge_duplicate_df(self):
        entity: Entity
        for entity in self.semantic_header.entities:
            if entity.merge_duplicate_df:
                self.connection.exec_query(CypherQueryLibrary.get_merge_duplicate_df_entity_query, **{"entity": entity})
                self.perf.finished_step(
                    activity=f"Merged duplicate [:DF] edges for (:{entity.get_label_string()}) done")

    def delete_parallel_dfs_derived(self):
        reified_entity: Entity
        original_entity: Entity
        relation: Relationship
        for reified_entity in self.semantic_header.get_entities_constructed_by_relations():
            if reified_entity.delete_parallel_df:
                relation = reified_entity.constructed_by.relation
                parent_entity = self.semantic_header.get_entity(relation.from_node.node_label)
                child_entity = self.semantic_header.get_entity(relation.to_node.node_label)
                for original_entity in [parent_entity, child_entity]:
                    self.connection.exec_query(CypherQueryLibrary.delete_parallel_directly_follows_derived,
                                               **{
                                                   "reified_entity": reified_entity,
                                                   "original_entity": original_entity
                                               })
                    self._write_message_to_performance(
                        f"Deleted parallel DF of (:{reified_entity.get_label_string()}) and (:"
                        f"{original_entity.get_label_string()})")

    def create_classes(self):
        classes = self.semantic_header.classes
        for _class in classes:
            self.connection.exec_query(CypherQueryLibrary.get_create_class_query, **{"_class": _class})
            self.connection.exec_query(CypherQueryLibrary.get_link_event_to_class_query,
                                       **{"_class": _class, "batch_size": self.batch_size})

    def create_static_nodes_and_relations(self):
        self._write_message_to_performance("No implementation yet")

    def add_attributes_to_classifier(self, relation, label, properties, copy_as):
        self.connection.exec_query(CypherQueryLibrary.add_attributes_to_classifier,
                                   **{
                                       "relation": relation, "label": label, "properties": properties,
                                       "copy_as": copy_as
                                   })
