import json
from abc import ABC
from pathlib import Path
from string import Template
from typing import List, Any, Optional, Union, Dict

from dataclasses import dataclass

from .interpreters import Interpreter
from ..utilities.auxiliary_functions import replace_undefined_value, create_list, get_id_attribute_from_label
import re

relation_directions = {
    "left-to-right": {"from_node": 0, "to_node": 1},
    "right-to-left": {"from_node": 1, "to_node": 0}
}


def get_node_or_rel_pattern(name, obj_type, condition, properties):
    if obj_type != "":
        pattern = f"{name}:{obj_type}"
    else:
        pattern = name
    if condition != "":
        pattern = f"{pattern} WHERE {condition}"
    elif properties != "":
        pattern = f"{pattern} {{{properties}}}"
    return pattern


def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass


@dataclass
class Class:
    label: str
    aggregate_from_nodes: str
    class_identifiers: List[str]
    include_identifier_in_label: bool
    ids: List[str]
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["Class"]:
        if obj is None:
            return None
        _label = obj.get("label")
        _aggregate_from_nodes = obj.get("aggregate_from_nodes")
        _class_identifiers = obj.get("class_identifiers")
        _include_identifier_in_label = replace_undefined_value(obj.get("include_identifier_in_label"), False)
        _ids = obj.get("ids")
        _query_interpreter = interpreter.class_qi
        return Class(_label, _aggregate_from_nodes, _class_identifiers, _include_identifier_in_label, _ids,
                     _query_interpreter)

    def get_condition(self, node_name="e"):
        return self.qi.get_condition(class_identifiers=self.class_identifiers, node_name=node_name)

    def get_group_by_statement(self, node_name="e"):
        return self.qi.get_group_by_statement(class_identifiers=self.class_identifiers,
                                              node_name=node_name)

    def get_class_properties(self) -> str:
        return self.qi.get_class_properties(class_identifiers=self.class_identifiers)

    def get_link_condition(self, class_node_name="c", event_node_name="e"):
        return self.qi.get_link_condition(class_identifiers=self.class_identifiers,
                                          class_node_name=class_node_name,
                                          event_node_name=event_node_name)

    def get_class_label(self):
        return self.qi.get_class_label(class_label=self.label, class_identifiers=self.class_identifiers,
                                       include_identifier_in_label=self.include_identifier_in_label)


@dataclass
class Condition:
    attribute: str
    values: List[Any]
    qi: Any

    @staticmethod
    def from_dict(obj: Any, query_interpreter) -> Optional["Condition"]:
        if obj is None:
            return None

        not_exist_properties = query_interpreter.get_not_exist_properties()
        _attribute = obj.get("attribute")
        _include_values = replace_undefined_value(obj.get("values"), not_exist_properties)
        _query_interpreter = query_interpreter
        return Condition(_attribute, _include_values, query_interpreter)


@dataclass()
class Attribute(ABC):
    optional: bool
    name: str
    dtype: str

    @staticmethod
    def from_string(attribute_str: str):
        _optional = False
        if "OPTIONAL" in attribute_str:
            _optional = True
            attribute_str = attribute_str.replace("OPTIONAL", "")
            attribute_str = attribute_str.strip()
        _name = attribute_str.split(" ")[0]
        _dtype = attribute_str.split(" ")[1]

        return Attribute(_optional, _name, _dtype)

    def get_pattern(self):
        if self.optional:
            pattern = "OPTIONAL $name $dtype"
        else:
            pattern = "$name $dtype"

        pattern = Template(pattern).substitute(name=self.name,
                                               dtype=self.dtype)

        return pattern

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class Attributes(ABC):
    closed: bool
    attributes: Dict[str, "Attribute"]

    @staticmethod
    def from_string(attributes_str: str):
        attributes_str = attributes_str.replace("}", "")
        attributes_str = attributes_str.replace(", ", ",")
        _closed = True
        if "OPEN" in attributes_str:
            _closed = False
            attributes_str = attributes_str.replace("OPEN", "")
            attributes_str = attributes_str.strip()

        attributes_str = attributes_str.split(",")
        _attributes = [Attribute.from_string(attribute_str) for attribute_str in attributes_str]
        _attributes = {attribute.name: attribute for attribute in _attributes}

        return Attributes(_closed, _attributes)

    def get_pattern(self):
        if self.attributes is None:
            return ""
        attribute_patterns = [attribute.get_pattern() for attribute in self.attributes.values()]
        attribute_patterns = ", ".join(attribute_patterns)
        if self.closed:
            pattern = "$attribute_patterns"
        else:
            pattern = "OPEN $attribute_patterns"

        pattern = Template(pattern).substitute(attribute_patterns=attribute_patterns)
        return pattern

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class Node(ABC):
    name: str
    str_types: List[str]
    types: List["Node"]
    # TODO make optional
    labels: List[str]
    attributes: "Attributes"
    closed: bool

    @staticmethod
    def from_string(node_description: str) -> Optional["Node"]:
        # we expect a node to be described in (node_name:Node_label)
        node_description = re.sub(r"[()]", "", node_description)
        node_components = node_description.split(":", 1)
        _name = node_components[0]
        label_str = ""
        _attributes = None

        if len(node_components) > 1:
            node_labels_attr = node_components[1]
            node_labels_attr = node_labels_attr.replace("'", "\"")
            if "{" in node_labels_attr:
                label_str = node_labels_attr.split(" {")[0]
                attributes_str = node_labels_attr.split(" {")[1]
                _attributes = Attributes.from_string(attributes_str)
            else:
                label_str = node_labels_attr

        _closed = True
        if "OPEN" in label_str:
            _closed = False
            label_str = label_str.replace("OPEN", "")
            label_str = label_str.strip()

        _labels = label_str.split(":")
        _types = [label for label in _labels if "Node" in label]
        _labels = [label for label in _labels if label not in set(_types)]

        return Node(name=_name, str_types=_types, types=[], labels=_labels,
                    attributes=_attributes, closed=_closed)

    def get_pattern(self, include_attributes=True):
        types_and_labels = self.str_types + self.labels
        if len(types_and_labels) > 0:
            node_label_str = ":".join(types_and_labels)
            node_pattern_str = "$node_name:$node_labels"
            node_pattern = Template(node_pattern_str).substitute(node_name=self.name,
                                                                 node_labels=node_label_str)
        else:
            node_pattern_str = "$node_name"
            node_pattern = Template(node_pattern_str).substitute(node_name=self.name)

        if not self.closed:
            node_pattern_str = "$node_pattern OPEN"
            node_pattern = Template(node_pattern_str).substitute(node_pattern=node_pattern)

        if self.attributes is not None and include_attributes:
            attributes_pattern = self.attributes.get_pattern()
            node_pattern_str = "$node_pattern {$attributes_pattern}"
            node_pattern = Template(node_pattern_str).substitute(node_pattern=node_pattern,
                                                                 attributes_pattern=attributes_pattern)

        node_pattern_str = "($node_pattern)"
        node_pattern = Template(node_pattern_str).substitute(node_pattern=node_pattern)

        return node_pattern

    def get_labels(self):
        labels = self.labels
        for node in self.types:
            labels.extend(node.get_labels())
        return list(set(labels))

    def get_attributes(self, include_optional=False):
        attributes = []
        if self.attributes is not None:
            if include_optional:
                attributes.extend(list(self.attributes.attributes.keys()))
            else:
                attributes.extend(
                    [key for key, attribute in self.attributes.attributes.items() if not attribute.optional])

        for node in self.types:
            attributes.extend(node.get_attributes())
        return list(set(attributes))

    def get_query(self, name, use_attributes=True):
        labels = self.get_labels()
        label_str = ":".join(labels)
        attributes = self.get_attributes()
        attributes = [f"{name}.{attribute} IS NOT NULL" for attribute in attributes]
        attribute_str = " AND ".join(attributes)

        if attribute_str != "" and use_attributes:
            return f"MATCH ({name}:{label_str} WHERE {attribute_str})"
        else:
            return f"MATCH ({name}:{label_str})"

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class NodeWithProperties:
    name: str
    obj_type: str
    object: Optional[Node]
    condition: str
    properties: str
    keywords: str

    @staticmethod
    def from_string(node_description: str, is_consequent=False):
        # we expect a node to be described in KEYWORDS (node_name:node_type {properties}) OR
        # (node_name: node_type WHERE properties)
        keywords = ""
        if "(" != node_description[0]:
            keywords = node_description.split(" (")[0]
            node_description = node_description.split(" (")[1]

        node_description = re.sub(r"[()]", "", node_description)
        index_first_colon = node_description.find(":")
        index_first_bracket = node_description.find("{")
        if index_first_bracket > index_first_colon or index_first_bracket == -1:
            node_components = node_description.split(":", 1)
        else:
            node_components = [node_description]

        _node_type = ""
        _properties_str = ""
        _condition_str = ""

        if len(node_components) > 1:
            _name = node_components[0]
            _node_type, _properties_str, _condition_str = NodeWithProperties.split_remaining_part(
                node_components[1])
        else:
            _name, _properties_str, _condition_str = NodeWithProperties.split_remaining_part(node_components[0])
            if "Node" in _name:
                _node_type = _name
                _name = ""

        if _name == "":
            _name = re.sub(r'(?<!^)(?=[A-Z])', '_', _node_type).lower()

        return NodeWithProperties(name=_name, obj_type=_node_type, object=None, condition=_condition_str,
                                  properties=_properties_str, keywords=keywords)

    @staticmethod
    def split_remaining_part(type_properties_condition: str):
        type_properties_condition = type_properties_condition.replace("'", "\"")
        _properties_str = ""
        _condition_str = ""
        if "{" in type_properties_condition:
            _name = type_properties_condition.split(" {")[0]
            _properties_str = type_properties_condition.split(" {")[1]
            _properties_str = _properties_str.replace("}", "")
        elif "WHERE" in type_properties_condition:
            _name = type_properties_condition.split(" WHERE")[0]
            _condition_str = type_properties_condition.split(" WHERE")[1]
        else:
            _name = type_properties_condition

        return _name, _properties_str, _condition_str

    def get_pattern(self):
        pattern = get_node_or_rel_pattern(name=self.name, obj_type=self.obj_type, condition=self.condition,
                                          properties=self.properties)
        pattern = f"({pattern})"

        return pattern

    def get_query(self, is_consequent, use_attributes=True):
        query_str = ""
        if not is_consequent:
            query_str += self.object.get_query(self.name, use_attributes=use_attributes)
            if self.condition != "":
                query_str += "\n" if query_str != "" else ""
                query_str += f"MATCH ({self.name} WHERE {self.condition})"
            if self.properties != "":
                query_str += "\n" if query_str != "" else ""
                query_str += f"MATCH ({self.name} {{{self.properties}}})"
        else:
            if self.obj_type != "":
                labels = self.object.get_labels()
                labels = ":".join(labels)

                if self.properties != "":
                    query_str += f"{self.keywords} ({self.name}:{labels} {{{self.properties}}})"
                else:
                    query_str += f"{self.keywords} ({self.name}:{labels})"
            else:
                if self.properties != "":
                    query_str += f"{self.keywords} ({self.name} {{{self.properties}}})"
                else:
                    query_str += f"{self.keywords} ({self.name})"
        return query_str

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class Relationship(ABC):
    name: str
    label: Optional[str]
    str_types: List[str]
    types: List["Relationship"]
    from_node: Optional[Node]
    from_node_name: str
    to_node: Optional[Node]
    to_node_name: str
    attributes: "Attributes"

    @staticmethod
    def from_string(relation_description: str) -> Optional["Relationship"]:
        # we expect a node to be described in (node_name:Node_label)
        nodes = re.findall(r'\([^<>]*\)', relation_description)
        _relation_string = re.findall(r'\[[^<>]*]', relation_description)[0]
        _relation_string = re.sub(r"[\[\]]", "", _relation_string)
        _relation_components = _relation_string.split(":")
        _name = _relation_components[0]
        label_str = ""
        _attributes = None

        if len(_relation_components) > 1:
            relation_label_attr = _relation_components[1]
            relation_label_attr = relation_label_attr.replace("'", "\"")
            if "{" in relation_label_attr:
                label_str = relation_label_attr.split(" {")[0]
                attributes_str = relation_label_attr.split(" {")[1]
                _attributes = Attributes.from_string(attributes_str)
            else:
                label_str = relation_label_attr

        if ">" in relation_description:
            direction = "left-to-right"
        elif "<" in relation_description:
            direction = "right-to-left"
        else:
            raise ValueError(f"In {relation_directions} no direction has been defined")

        _from_node_name = nodes[relation_directions[direction]["from_node"]]
        _to_node_name = nodes[relation_directions[direction]["to_node"]]
        _from_node_name = re.sub(r"[()]", "", _from_node_name)
        _to_node_name = re.sub(r"[()]", "", _to_node_name)

        _labels = label_str.split(":")
        _types = [label for label in _labels if "Relation" in label]
        _labels = [label for label in _labels if label not in set(_types)]
        if len(_labels) > 1:
            raise ValueError(f"Too many labels for {relation_description} defined")
        if len(_labels) == 1:
            _label_str = _labels[0]
        else:
            _label_str = None

        return Relationship(name=_name, label=_label_str,
                            str_types=_types, types=[],
                            from_node_name=_from_node_name, to_node_name=_to_node_name,
                            from_node=None, to_node=None,
                            attributes=_attributes)

    def get_pattern(self, include_attributes=True):
        types_and_label = []
        types_and_label.extend(self.str_types)
        if self.label is not None:
            types_and_label.append(self.label)

        if len(types_and_label) > 0:
            rel_label_str = ":".join(types_and_label)
            rel_pattern_str = "$rel_name:$rel_labels"
            rel_pattern = Template(rel_pattern_str).substitute(rel_name=self.name,
                                                               rel_labels=rel_label_str
                                                               )
        else:
            rel_pattern_str = "$rel_name"
            rel_pattern = Template(rel_pattern_str).substitute(rel_name=self.name)

        if self.attributes is not None and include_attributes:
            attributes_pattern = self.attributes.get_pattern()
            rel_pattern_str = "$rel_pattern {$attributes_pattern}"
            rel_pattern = Template(rel_pattern_str).substitute(rel_pattern=rel_pattern,
                                                               attributes_pattern=attributes_pattern)

        from_node_pattern = self.from_node.get_pattern(include_attributes=False)
        to_node_pattern = self.to_node.get_pattern(include_attributes=False)

        rel_pattern_str = "$from_node - [$rel_pattern] -> $to_node"
        rel_pattern = Template(rel_pattern_str).substitute(from_node=from_node_pattern,
                                                           rel_pattern=rel_pattern,
                                                           to_node=to_node_pattern)

        return rel_pattern

    def get_label(self):
        labels = [self.label] if self.label is not None else []
        for relationship in self.types:
            labels.append(relationship.get_label())
        labels = list(set(labels))
        if len(labels) > 1:
            raise ValueError(f"Too many labels defined for relation {self.name}")
        if len(labels) == 1:
            return labels[0]
        return ""

    def get_attributes(self, include_optional=False):
        attributes = []
        if self.attributes is not None:
            if include_optional:
                attributes.extend(list(self.attributes.attributes.keys()))
            else:
                attributes.extend(
                    [key for key, attribute in self.attributes.attributes.items() if not attribute.optional])

        for node in self.types:
            attributes.extend(node.get_attributes())
        return list(set(attributes))

    def get_query(self, rel_name, from_node_name, to_node_name):
        query_str = ""

        label = self.get_label()
        attributes = self.get_attributes()
        attributes = [f"{rel_name}.{attribute} IS NOT NULL" for attribute in attributes]
        attribute_str = " AND ".join(attributes)

        from_node_label_str = ":".join(self.from_node.get_labels())
        if from_node_label_str != "":
            from_node = f"{from_node_name}:{from_node_label_str}"
        else:
            from_node = from_node_name

        to_node_label_str = ":".join(self.to_node.get_labels())
        if to_node_label_str != "":
            to_node = f"{to_node_name}:{to_node_label_str}"
        else:
            to_node = to_node_name

        if attribute_str != "":
            query_str += f"MATCH ({from_node}) - [{rel_name}:{label} WHERE {attribute_str}] -> ({to_node})"
        else:
            query_str += f"MATCH ({from_node}) - [{rel_name}:{label}] -> ({to_node})"

        return query_str

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class RelationshipWithProperties:
    name: str
    obj_type: str
    object: Optional[Relationship]
    from_node: NodeWithProperties
    to_node: NodeWithProperties
    condition: str
    properties: str
    keywords: str

    @staticmethod
    def from_string(rel_description: str, is_consequent=False):
        # we expect a node to be described in KEYWORDS (node_name:node_type {properties}) OR
        # (node_name: node_type WHERE properties)
        keywords = ""
        if "(" != rel_description[0]:
            keywords = rel_description.split(" (")[0]
            rel_description = rel_description.replace(f"{keywords} ", "")

        nodes = re.findall(r'\([^<>]*\)', rel_description)
        _relation_string = re.findall(r'\[[^<>]*]', rel_description)[0]
        _relation_string = re.sub(r"[\[\]]", "", _relation_string)
        index_first_colon = _relation_string.find(":")
        index_first_bracket = _relation_string.find("{")
        if index_first_bracket > index_first_colon or index_first_bracket == -1:
            rel_components = _relation_string.split(":", 1)
        else:
            rel_components = [_relation_string]

        if ">" in rel_description:
            direction = "left-to-right"
        elif "<" in rel_description:
            direction = "right-to-left"
        else:
            raise ValueError(f"In {relation_directions} no direction has been defined")

        _obj_type = ""
        _properties_str = ""
        _condition_str = ""
        _from_node_description = nodes[relation_directions[direction]["from_node"]]
        _from_node_with_properties = NodeWithProperties.from_string(_from_node_description, is_consequent)
        _to_node_description = nodes[relation_directions[direction]["to_node"]]
        _to_node_with_properties = NodeWithProperties.from_string(_to_node_description, is_consequent)

        if not is_consequent:
            try:
                _name = rel_components[0]
                _obj_type, _properties_str, _condition_str = NodeWithProperties.split_remaining_part(
                    rel_components[1])
            except ValueError:
                raise ValueError("No type of node has been defined")
        else:
            if len(rel_components) > 1:
                _name = rel_components[0]
                _obj_type, _properties_str, _condition_str = RelationshipWithProperties.split_remaining_part(
                    rel_components[1])
            else:
                _name, _properties_str, _condition_str = NodeWithProperties.split_remaining_part(rel_components[0])

        return RelationshipWithProperties(name=_name,
                                          obj_type=_obj_type, from_node=_from_node_with_properties,
                                          to_node=_to_node_with_properties,
                                          object=None, condition=_condition_str,
                                          properties=_properties_str, keywords=keywords)

    @staticmethod
    def split_remaining_part(type_properties_condition: str):
        type_properties_condition = type_properties_condition.replace("'", "\"")
        _properties_str = ""
        _condition_str = ""
        if "{" in type_properties_condition:
            _name = type_properties_condition.split(" {")[0]
            _properties_str = type_properties_condition.split(" {")[1]
            _properties_str = _properties_str.replace("}", "")
        elif "WHERE" in type_properties_condition:
            _name = type_properties_condition.split(" WHERE")[0]
            _condition_str = type_properties_condition.split(" WHERE")[1]
        else:
            _name = type_properties_condition

        return _name, _properties_str, _condition_str

    def get_pattern(self):

        from_node_pattern = self.from_node.get_pattern()
        to_node_pattern = self.to_node.get_pattern()

        pattern = get_node_or_rel_pattern(name=self.name, obj_type=self.obj_type, condition=self.condition,
                                          properties=self.properties)

        pattern = f"{from_node_pattern} - [{pattern}] -> {to_node_pattern}"

        return pattern

    def get_query(self, is_consequent):
        query_str = ""
        if not is_consequent:
            query_str += self.object.get_query(self.name, self.from_node.name, self.to_node.name)
            from_query = self.from_node.get_query(is_consequent, use_attributes=False)
            if from_query != "":
                query_str += "\n" if query_str != "" else ""
                query_str += from_query
            to_query = self.to_node.get_query(is_consequent, use_attributes=False)
            if to_query != "":
                query_str += "\n" if query_str != "" else ""
                query_str += to_query
            if self.condition != "":
                query_str += "\n" if query_str != "" else ""
                query_str += f"MATCH () - [{self.name} WHERE {self.condition}] -> ()"
            if self.properties != "":
                query_str += "\n" if query_str != "" else ""
                query_str += f"MATCH () - [{self.name} {{{self.properties}}}] -> ()"
        else:
            if self.obj_type != "":
                label = self.object.get_label()
                rel_pattern = f"{self.name}:{label}"
            else:
                rel_pattern = self.name

            if self.properties != "":
                rel_pattern = f"{rel_pattern} {{{self.properties}}}"

            query_str += f"{self.keywords} ({self.from_node.name}) - [{rel_pattern}] -> ({self.to_node.name})"
        return query_str

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class Proposition:
    proposition: Union["NodeWithProperties", "RelationshipWithProperties"]
    is_consequent: bool

    @staticmethod
    def from_string(proposition_str, is_consequent=False):
        if "-" in proposition_str:
            _proposition = RelationshipWithProperties.from_string(proposition_str, is_consequent)
        else:
            _proposition = NodeWithProperties.from_string(proposition_str, is_consequent)
        return Proposition(proposition=_proposition, is_consequent=is_consequent)

    @staticmethod
    def from_dict(obj: Any, is_consequent):
        return Proposition.from_string(obj, is_consequent)

    def get_pattern(self):
        return self.proposition.get_pattern()

    def get_query(self):
        return self.proposition.get_query(self.is_consequent)

    def __repr__(self):
        return self.get_pattern()


@dataclass()
class Constructor(ABC):
    antecedents: List[Proposition]
    consequents: List[Proposition]

    @staticmethod
    def from_dict(obj: Any):
        if obj is None:
            return None
        _antecedents = create_list(Proposition, obj.get("antecedents"), False)
        _consequents = create_list(Proposition, obj.get("consequents"), True)
        return Constructor(antecedents=_antecedents, consequents=_consequents)

    def get_query(self):
        query_str = ""
        for antecedent in self.antecedents:
            query_str += "\n" if query_str != "" else ""
            query_str += antecedent.get_query()
        for consequent in self.consequents:
            query_str += "\n" if query_str != "" else ""
            query_str += consequent.get_query()
        return query_str


@dataclass()
class NodesWithConstructors(ABC):
    nodes: Dict[str, "Node"]
    constructors: Dict[str, "Constructor"]

    @staticmethod
    def from_list(list_with_obj: List[Any]):
        _nodes = {}
        _constructors = {}
        for obj in list_with_obj:
            node: Node
            node = Node.from_string(obj.get("node_description"))
            constructor = Constructor.from_dict(obj.get("constructor"))
            _nodes[node.name] = node
            if constructor is not None:
                _constructors[node.name] = constructor

        nodes_with_constructors = NodesWithConstructors(nodes=_nodes, constructors=_constructors)
        nodes_with_constructors.link_nodes()

        return nodes_with_constructors

    def link_nodes(self):
        for name, node in self.nodes.items():
            for type_str in node.str_types:
                node.types.append(self.nodes[type_str])


@dataclass()
class RelationshipsWithConstructors(ABC):
    relationships: Dict[str, "Relationship"]
    constructors: Dict[str, "Constructor"]

    @staticmethod
    def from_list(list_with_obj: List[Any]):
        _relationships = {}
        _constructors = {}
        for obj in list_with_obj:
            relationship = Relationship.from_string(obj.get("relationship_description"))
            constructor = Constructor.from_dict(obj.get("constructor"))
            _relationships[relationship.name] = relationship
            if constructor is not None:
                _constructors[relationship.name] = constructor

        relationship_with_constructors = RelationshipsWithConstructors(relationships=_relationships,
                                                                       constructors=_constructors)
        relationship_with_constructors.link_relationships()

        return relationship_with_constructors

    def link_relationships(self):
        for name, relationship in self.relationships.items():
            for type_str in relationship.str_types:
                relationship.types.append(self.relationships[type_str])


@dataclass
class RelationConstructorByNodes(ABC):
    from_node_label: str
    to_node_label: str
    foreign_key: str
    primary_key: str
    reversed: bool
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["RelationConstructorByNodes"]:
        if obj is None:
            return None

        _from_node_label = obj.get("from_node_label")
        _to_node_label = obj.get("to_node_label")
        _foreign_key = obj.get("foreign_key")
        _primary_key = replace_undefined_value(obj.get("primary_key"), "ID")
        _reversed = replace_undefined_value(obj.get("reversed"), False)
        return RelationConstructorByNodes(from_node_label=_from_node_label, to_node_label=_to_node_label,
                                          foreign_key=_foreign_key, primary_key=_primary_key,
                                          reversed=_reversed, qi=interpreter.relation_constructor_by_nodes_qi)

    def get_id_attribute_from_from_node(self):
        return get_id_attribute_from_label(self.from_node_label)

    def get_id_attribute_from_to_node(self):
        return get_id_attribute_from_label(self.to_node_label)


class RelationshipOrNode(ABC):
    @staticmethod
    def from_string(relation_description: str, interpreter: Interpreter) -> Union["Relationship", "Node"]:
        if "-" in relation_description:
            return Relationship.from_string(relation_description, interpreter)
        else:
            return Node.from_string(relation_description, interpreter)


@dataclass
class RelationConstructorByRelations(ABC):
    antecedents: List[Relationship]
    consequent: Relationship
    from_node_name: str
    to_node_name: str
    from_node_label: str
    to_node_label: str
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> \
            Optional["RelationConstructorByRelations"]:
        if obj is None:
            return None

        _antecedents = [RelationshipOrNode.from_string(y, interpreter) for y in obj.get("antecedents")]
        _consequent = Relationship.from_string(obj.get("consequent"), interpreter)

        _from_node_name = _consequent.from_node.name
        _to_node_name = _consequent.to_node.name
        _from_node_label = _consequent.from_node.label
        _to_node_label = _consequent.to_node.label

        return RelationConstructorByRelations(antecedents=_antecedents, consequent=_consequent,
                                              from_node_name=_from_node_name,
                                              to_node_name=_to_node_name, from_node_label=_from_node_label,
                                              to_node_label=_to_node_label,
                                              qi=interpreter.relation_constructor_by_relations_qi)

    def get_from_node_name(self):
        return self.consequent.from_node.name

    def get_to_node_name(self):
        return self.consequent.to_node.name

    def get_from_node_label(self):
        return self.consequent.from_node.label

    def get_to_node_label(self):
        return self.consequent.to_node.label

    def get_id_attribute_from_from_node(self):
        return get_id_attribute_from_label(self.from_node_label)

    def get_id_attribute_from_to_node(self):
        return get_id_attribute_from_label(self.to_node_label)

    def get_antecedent_query(self):
        return self.qi.get_antecedent_query(antecedents=self.antecedents)


@dataclass
class RelationConstructorByQuery(ABC):
    query: str
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["RelationConstructorByQuery"]:
        if obj is None:
            return None

        _query = obj.get("query")

        return RelationConstructorByQuery(query=_query, qi=interpreter.relation_constructor_by_query_qi)


@dataclass
class Relation(ABC):
    include: bool
    type: str
    constructed_by: Union[RelationConstructorByNodes, RelationConstructorByRelations, RelationConstructorByQuery]
    constructor_type: str
    include_properties: bool
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["Relation"]:
        if obj is None:
            return None
        _include = replace_undefined_value(obj.get("include"), True)
        if not _include:
            return None

        _type = obj.get("type")

        _constructed_by = RelationConstructorByNodes.from_dict(obj.get("constructed_by_nodes"),
                                                               interpreter)
        if _constructed_by is None:
            _constructed_by = RelationConstructorByRelations.from_dict(obj.get("constructed_by_relations"),
                                                                       interpreter)
        if _constructed_by is None:
            _constructed_by = RelationConstructorByQuery.from_dict(obj.get("constructed_by_query"), interpreter)

        _constructor_type = _constructed_by.__class__.__name__

        _include_properties = replace_undefined_value(obj.get("include_properties"), True)

        return Relation(_include, _type, constructed_by=_constructed_by, constructor_type=_constructor_type,
                        include_properties=_include_properties,
                        qi=interpreter.relation_qi)


@dataclass
class EntityConstructorByNode(ABC):
    node_label: str
    conditions: List[Condition]
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["EntityConstructorByNode"]:
        if obj is None:
            return None

        _node_label = obj.get("node_label")
        _conditions = create_list(Condition, obj.get("conditions"), interpreter.condition_qi)

        return EntityConstructorByNode(node_label=_node_label, conditions=_conditions,
                                       qi=interpreter.entity_constructor_by_nodes_qi)


@dataclass
class EntityConstructorByRelation(ABC):
    relation: Relationship
    conditions: List[Condition]
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter = Interpreter) -> \
            Optional["EntityConstructorByRelation"]:
        if obj is None:
            return None

        _relation = Relationship.from_string(obj.get("relation_type"), interpreter)
        _conditions = create_list(Condition, obj.get("conditions"), interpreter)

        return EntityConstructorByRelation(relation=_relation, conditions=_conditions,
                                           qi=interpreter.entity_constructor_by_relation_qi)

    def get_relation_type(self):
        return self.relation.relation_type


@dataclass
class EntityConstructorByQuery(ABC):
    query: str
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["EntityConstructorByQuery"]:
        if obj is None:
            return None

        _query = obj.get("query")

        return EntityConstructorByQuery(query=_query, qi=interpreter.entity_constructor_by_query_qi)


@dataclass
class Entity(ABC):
    include: bool
    constructed_by: Union[EntityConstructorByNode, EntityConstructorByRelation, EntityConstructorByQuery]
    constructor_type: str
    type: str
    labels: List[str]
    primary_keys: List[str]
    all_entity_attributes: List[str]
    entity_attributes_wo_primary_keys: List[str]
    corr: bool
    df: bool
    include_label_in_df: bool
    merge_duplicate_df: bool
    delete_parallel_df: bool
    qi: Any

    def get_primary_keys(self):
        return self.primary_keys

    @staticmethod
    def determine_labels(labels: List[str], _type: str) -> List[str]:
        if "Entity" in labels:
            labels.remove("Entity")

        if _type not in labels:
            labels.insert(0, _type)

        return labels

    def get_properties(self):
        properties = {}
        for condition in self.constructed_by.conditions:
            properties[condition.attribute] = condition.values

        return properties

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["Entity"]:

        if obj is None:
            return None
        _include = replace_undefined_value(obj.get("include"), True)
        if not _include:
            return None

        _constructed_by = EntityConstructorByNode.from_dict(obj.get("constructed_by_node"), interpreter=interpreter)
        if _constructed_by is None:
            _constructed_by = EntityConstructorByRelation.from_dict(obj.get("constructed_by_relation"),
                                                                    interpreter=interpreter)
        if _constructed_by is None:
            _constructed_by = EntityConstructorByQuery.from_dict(obj.get("constructed_by_query"),
                                                                 interpreter=interpreter)

        _constructor_type = _constructed_by.__class__.__name__
        _type = obj.get("type")
        _labels = replace_undefined_value(obj.get("labels"), [])
        _labels = Entity.determine_labels(_labels, _type)
        _primary_keys = obj.get("primary_keys")
        # entity attributes may have primary keys (or not)
        _entity_attributes = replace_undefined_value(obj.get("entity_attributes"), [])
        # create a list of all entity attributes
        if len(_primary_keys) > 1:  # more than 1 primary key, also store the primary keys separately
            _all_entity_attributes = list(set(_entity_attributes + _primary_keys))
        else:
            # remove the primary keys from the entity attributes
            _all_entity_attributes = list(set(_entity_attributes).difference(set(_primary_keys)))
        # remove the primary keys
        _entity_attributes_wo_primary_keys = [attr for attr in _all_entity_attributes if attr not in _primary_keys]

        _corr = _include and replace_undefined_value(obj.get("corr"), False)
        _df = _corr and replace_undefined_value(obj.get("df"), False)
        _include_label_in_df = _df and replace_undefined_value(obj.get("include_label_in_df"), False)
        _merge_duplicate_df = _df and replace_undefined_value(obj.get("merge_duplicate_df"), False)

        _delete_parallel_df = _df and obj.get("delete_parallel_df")

        return Entity(include=_include, constructed_by=_constructed_by, constructor_type=_constructor_type,
                      type=_type, labels=_labels, primary_keys=_primary_keys,
                      all_entity_attributes=_all_entity_attributes,
                      entity_attributes_wo_primary_keys=_entity_attributes_wo_primary_keys,
                      corr=_corr, df=_df, include_label_in_df=_include_label_in_df,
                      merge_duplicate_df=_merge_duplicate_df,
                      delete_parallel_df=_delete_parallel_df,
                      qi=interpreter.entity_qi)

    def get_label_string(self):
        return self.qi.get_label_string(self.labels)

    def get_labels(self):
        return ["Entity"] + self.labels

    def get_df_label(self):
        return self.qi.get_df_label(self.include_label_in_df, self.type)

    def get_composed_primary_id(self, node_name: str = "e"):
        return self.qi.get_composed_primary_id(self.primary_keys, node_name)

    def get_entity_attributes(self, node_name: str = "e"):
        return self.qi.get_entity_attributes(self.primary_keys, self.entity_attributes_wo_primary_keys,
                                             node_name)

    def get_entity_attributes_as_node_properties(self):
        if len(self.all_entity_attributes) > 0:
            return self.qi.get_entity_attributes_as_node_properties(self.all_entity_attributes)
        else:
            return ""

    def get_primary_key_existing_condition(self, node_name: str = "e"):
        return self.qi.get_primary_key_existing_conditionge(self.primary_keys, node_name)

    def create_condition(self, name: str) -> str:
        return self.qi.create_condition(self.constructed_by.conditions, name)

    def get_where_condition(self, node_name: str = "e"):
        return self.qi.get_where_condition(self.constructed_by.conditions, self.primary_keys, node_name)

    def get_where_condition_correlation(self, node_name: str = "e", node_name_id: str = "n"):
        return self.qi.get_where_condition_correlation(self.constructed_by.conditions, self.primary_keys,
                                                       node_name, node_name_id)


@dataclass
class Log:
    include: bool
    has: bool
    qi: Any

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> "Log":
        if obj is None:
            return Log(False, False, interpreter.log_qi)
        _include = replace_undefined_value(obj.get("include"), True)
        if not _include:
            return Log(False, False, interpreter.log_qi)
        _has = replace_undefined_value(obj.get("has"), True)
        return Log(_include, _has, qi=interpreter.log_qi)


class SemanticHeader(ABC):
    def __init__(self, name: str, version: str,
                 nodes: Dict[str, Node], relations: Dict[str, Relationship],
                 constructors: Dict[str, Constructor],
                 classes: List[Class], log: Log):
        self.name = name
        self.version = version
        self.nodes = nodes
        self.relations = relations
        self.constructors = constructors
        self.classes = classes
        self.log = log

    def get_entity(self, entity_name) -> Optional[Node]:
        for node in self.nodes:
            if "Entity" in node.labels:
                if entity_name == node.name:
                    return node
        return None

    def get_node(self, node_name) -> Optional[Node]:
        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise ValueError(f"{node_name} is not defined")

    @staticmethod
    def from_dict(obj: Any, interpreter: Interpreter) -> Optional["SemanticHeader"]:
        if obj is None:
            return None
        _name = obj.get("name")
        _version = obj.get("version")
        _nodes_with_constructors = NodesWithConstructors.from_list(obj.get("nodes"))
        _nodes = _nodes_with_constructors.nodes
        _constructors = _nodes_with_constructors.constructors
        _relationships_with_constructors = RelationshipsWithConstructors.from_list(obj.get("relationships"))
        _relations = _relationships_with_constructors.relationships
        _rel_constructors = _relationships_with_constructors.constructors
        _constructors.update(_rel_constructors)
        _classes = create_list(Class, obj.get("classes"), interpreter)
        _log = Log.from_dict(obj.get("log"), interpreter)
        sh = SemanticHeader(_name, _version, _nodes, _relations, _constructors,
                            _classes, _log)
        sh.link_nodes_to_relationships()
        sh.link_constructors()
        return sh

    def link_nodes_to_relationships(self):
        for name, relationship in self.relations.items():
            from_node_name = relationship.from_node_name
            to_node_name = relationship.to_node_name
            relationship.from_node = self.nodes[from_node_name]
            relationship.to_node = self.nodes[to_node_name]

    def link_constructors(self):
        for name, constructor in self.constructors.items():
            antecedents = constructor.antecedents
            consequents = constructor.consequents
            for antecedent in antecedents:
                self.link_proposition(antecedent)
            for consequent in consequents:
                self.link_proposition(consequent)

    def link_proposition(self, proposition):

        obj_type = proposition.proposition.obj_type
        if "Node" in obj_type:
            proposition.proposition.object = self.nodes[obj_type]
        else:
            proposition.proposition.object = self.relations[obj_type]
            from_node_name = proposition.proposition.from_node.obj_type
            to_node_name = proposition.proposition.to_node.obj_type
            if "Node" in from_node_name:
                proposition.proposition.from_node.object = self.nodes[from_node_name]
            if "Node" in to_node_name:
                proposition.proposition.to_node.object = self.nodes[to_node_name]

    @staticmethod
    def create_semantic_header(path: Path, query_interpreter):
        with open(path) as f:
            json_semantic_header = json.load(f)

        semantic_header = SemanticHeader.from_dict(json_semantic_header, query_interpreter)
        return semantic_header

    def get_entities_constructed_by_nodes(self):
        return [entity for entity in self.entities if
                entity.constructor_type == "EntityConstructorByNode"]

    def get_entities_constructed_by_relations(self):
        return [entity for entity in self.entities if
                entity.constructor_type == "EntityConstructorByRelation"]

    def get_entities_constructed_by_query(self):
        return [entity for entity in self.entities if
                entity.constructor_type == "EntityConstructorByQuery"]

    def get_relations_derived_from_nodes(self):
        return [relation for relation in self.relations if
                relation.constructor_type == "RelationConstructorByNodes"]

    def get_relations_derived_from_relations(self):
        return [relation for relation in self.relations if
                "RelationConstructorByRelations" in relation.constructor_type]

    def get_relations_derived_from_query(self):
        return [relation for relation in self.relations if
                relation.constructor_type == "RelationConstructorByQuery"]
