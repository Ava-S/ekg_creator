from string import Template
from typing import List, Union


class ClassCypher:
    @staticmethod
    def get_condition(class_identifiers, node_name="e"):
        # reformat to where e.key is not null to create with condition
        return " AND ".join([f"{node_name}.{key} IS NOT NULL" for key in class_identifiers])

    @staticmethod
    def get_group_by_statement(class_identifiers, node_name="e"):
        # reformat to e.key with alias to create with condition
        return 'distinct ' + ' , '.join([f"{node_name}.{key} AS {key}" for key in class_identifiers])

    @staticmethod
    def get_class_properties(class_identifiers) -> str:
        ids = class_identifiers
        if "name" not in ids:
            ids = ["name"] + ids

        # create a combined id in string format
        _id = "+".join([f"{key}" for key in class_identifiers])
        # add class_identifiers to the keys if there are multiple
        if len(class_identifiers) > 1:
            required_keys = [_id] + class_identifiers
        else:
            required_keys = [_id]

        node_properties = ', '.join([f"{_id}: {key}" for _id, key in zip(ids, required_keys)])
        node_properties += f", aggregationType: '{_id}'"  # save ID also as string that captures the type

        return node_properties

    @staticmethod
    def get_link_condition(class_identifiers, class_node_name="c", event_node_name="e"):
        if len(class_identifiers) == 1:
            return f"{class_node_name}.name = {event_node_name}.{class_identifiers[0]}"
        return ' AND '.join([f"{class_node_name}.{key} = {event_node_name}.{key}" for key in class_identifiers])

    @staticmethod
    def get_class_label(class_label, class_identifiers, include_identifier_in_label):
        if len(class_identifiers) > 0 and include_identifier_in_label:
            return f"{class_label}:{class_label}_" + "_".join([f"{key}" for key in class_identifiers])
        else:
            return class_label


class LogCypher:
    pass


class ConditionCypher:
    @staticmethod
    def get_not_exist_properties():
        return ["IS NOT NULL", '<> "nan"', '<> "None"']

    @staticmethod
    def get_values(values):
        if values != ["IS NOT NULL", '<> "nan"', '<> "None"']:
            return [f'''= "{include_value}"''' for include_value in values]


class EntityCypher:
    @staticmethod
    def get_label_string(labels):
        return "Entity:" + ":".join(labels)

    @staticmethod
    def get_df_label(include_label_in_df, type):
        """
        Create the df label based on self.option_DF_entity_type_in_label
        If not in entity type, add it as property to the label
        @param entity:
        @return:
        """
        if include_label_in_df:
            return f'DF_{type.upper()}'
        else:
            return f'DF'

    @staticmethod
    def get_composed_primary_id(primary_keys, node_name: str = "e"):
        return "+\"-\"+".join([f"{node_name}.{key}" for key in primary_keys])

    @staticmethod
    def get_entity_attributes(primary_keys, entity_attributes_wo_primary_keys, node_name: str = "e"):
        # TODO: check what happens when entity does not exist
        primary_key_list = [f"{node_name}.{key} as {key}" for key in primary_keys]
        entity_attribute_list = [f"COLLECT(distinct {node_name}.{attr}) as {attr}" for attr in
                                 entity_attributes_wo_primary_keys]
        complete_list = primary_key_list + entity_attribute_list
        return ','.join(complete_list)

    @staticmethod
    def get_entity_attributes_as_node_properties(all_entity_attributes):
        return ',' + ',\n'.join([f"{key}: {key}" for key in all_entity_attributes])

    @staticmethod
    def get_primary_key_existing_condition(primary_keys, node_name: str = "e"):
        return " AND ".join(
            [f'''{node_name}.{key} IS NOT NULL AND {node_name}.{key} <> "nan" AND {node_name}.{key}<> "None"''' for key
             in primary_keys])

    @staticmethod
    def create_condition(conditions, name: str) -> str:
        """
        Converts a dictionary into a string that can be used in a WHERE statement to find the correct node/relation
        @param name: str, indicating the name of the node/rel
        @param properties: Dictionary containing of property name and value
        @return: String that can be used in a where statement
        """

        condition_list = []
        for condition in conditions:
            attribute_name = condition.attribute
            include_values = condition.values
            for value in include_values:
                condition_list.append(f'''{name}.{attribute_name} = "{value}"''')
        condition_string = " AND ".join(condition_list)
        return condition_string

    @staticmethod
    def get_where_condition(conditions, primary_keys, node_name: str = "e"):
        primary_key_existing_condition = EntityCypher.get_primary_key_existing_condition(primary_keys, node_name)
        extra_conditions = EntityCypher.create_condition(conditions, node_name)
        if extra_conditions != "":
            return f'''{primary_key_existing_condition} AND {extra_conditions}'''
        else:
            return primary_key_existing_condition

    @staticmethod
    def get_where_condition_correlation(conditions, primary_keys, node_name: str = "e", node_name_id: str = "n"):
        primary_key_condition = f"{EntityCypher.get_composed_primary_id(primary_keys, node_name)} = {node_name_id}.ID"
        extra_conditions = EntityCypher.create_condition(conditions, node_name)
        if extra_conditions != "":
            return f'''{primary_key_condition} AND {extra_conditions}'''
        else:
            return primary_key_condition


class RelationCypher:
    pass


class RelationConstructorByNodesCypher:
    pass


class RelationConstructorByRelationsCypher:
    @staticmethod
    def get_antecedent_query(antecedents: List[Union["Relationship", "Node"]]):
        from ekg_creator.data_managers.semantic_header import Relationship
        from ekg_creator.data_managers.semantic_header import Node
        if not all(isinstance(x, (Relationship, Node)) for x in antecedents):
            raise TypeError("Antecedents are not of type Relationship or Node")

        antecedents_query = [f"MATCH {antecedent.get_pattern()}" for antecedent in antecedents]
        antecedents_query = "\n".join(antecedents_query)

        return antecedents_query


class RelationConstructorByQueryCypher:
    pass


class EntityConstructorByQueryCypher:
    pass


class EntityConstructorByRelationCypher:
    pass


class EntityConstructorByNodesCypher:
    pass


class RelationshipCypher:
    @staticmethod
    def get_relationship_pattern(from_node, to_node, relation_name, relation_type, has_direction):
        from_node_pattern = from_node.get_pattern()
        to_node_pattern = to_node.get_pattern()
        if relation_type != "":
            relationship_pattern = "$from_node - [$relation_name:$relation_type] -> $to_node" if has_direction \
                else "$from_node - [$relation_name:$relation_type] - $to_node"
            relationship_pattern = Template(relationship_pattern).substitute(from_node=from_node_pattern,
                                                                             to_node=to_node_pattern,
                                                                             relation_name=relation_name,
                                                                             relation_type=relation_type)
        else:
            relationship_pattern = "$from_node - [$relation_name] -> $to_node" if has_direction \
                else "$from_node - [$relation_name] - $to_node"
            relationship_pattern = Template(relationship_pattern).substitute(from_node=from_node_pattern,
                                                                             to_node=to_node_pattern,
                                                                             relation_name=relation_name)
        return relationship_pattern
