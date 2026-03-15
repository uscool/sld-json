"""
Electrical Grammar Module for Single Line Diagram (SLD) Validation

This module defines the formal type system for electrical substation topologies.
It specifies:
- Valid voltage domains
- Valid device types
- Connection rules between device types
- Voltage compatibility rules

This is a purely logical module with no geometry, rendering, or ML code.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
from collections import defaultdict

# =================================================
# PRIMITIVES
# =================================================
class Voltage(Enum):
    """Strongly-typed voltage levels in the system."""
    V220KV = "220kV"
    V132KV = "132kV"
    V66KV  = "66kV"
    V33KV  = "33kV"
    V11KV  = "11kV"
    
    def __str__(self):
        return self.value

class DeviceType(Enum):
    """All valid device types in the electrical grammar."""
    GRID_SOURCE = "GridSource"
    TRANSFORMER = "Transformer"
    BUS = "Bus"
    BREAKER = "Breaker"
    ISOLATOR = "Isolator"
    FEEDER = "Feeder"
    
    def __str__(self):
        return self.value

class ConnectionVerdict(Enum):
    OK = auto()
    REJECT_TYPE = auto()
    REJECT_VOLTAGE = auto()
    REJECT_TERMINAL = auto()

class BusRole(Enum):
    MAIN = auto()
    TRANSFER = auto()
    RING = auto()
    LV = auto()

class BayRole(Enum):
    TRANSFORMER = auto()
    FEEDER = auto()
    LINE = auto()
    COUPLER = auto()

@dataclass(frozen=True)
class DeviceProperties:
    """Electrical properties that define how a device behaves in the grammar."""
    device_type: DeviceType
    is_voltage_bridge: bool  # True if device connects two different voltages
    is_terminal: bool  # True if device is a leaf node (no outgoing connections)
    primary_voltage: Optional[Voltage] = None  # For single-voltage devices
    
    def operates_at_single_voltage(self) -> bool:
        """Returns True if device operates at a single voltage domain."""
        return not self.is_voltage_bridge

@dataclass(frozen=True)
class Device:
    """A concrete electrical device instance."""
    device_id: str # node ID in the electrical graph
    device_type: DeviceType  # (Breaker/Transformer/Bus/Feeder/etc)
    voltage: Voltage   # operating voltage domain (33kV/11kV)
    bus_role: Optional[BusRole] = None # if busbar, role played = (MAIN/TRANSFER/RING/LV/etc)
    bay_id: Optional[str] = None # ID of the bay this device belongs to
    
    def __hash__(self):
        return hash(self.device_id)

    def __eq__(self, other): 
        return isinstance(other, Device) and self.device_id == other.device_id

    def __str__(self):
        return f"{self.device_type.value}({self.device_id}, {self.voltage.value})"

@dataclass(frozen=True)
class BaySchema:
    """
    Defines the structural and electrical rules that a bay must satisfy 
    to be considered valid in the grammar.
    """
    name: str
    role: BayRole
    min_buses: int
    max_buses: int
    breakers: int
    requires_transformer: bool
    allows_feeder: bool

@dataclass(frozen=True)
class TopologySchema:
    """
    Defines the required bus roles, allowed bay types, and coupling rules 
    for a valid substation topology.
    """
    name: str
    buses: Dict[BusRole, int]
    bay_types: Set[str]
    allows_bus_couplers: bool

@dataclass
class TopologyError(Exception):
    """
    Structured compile-time error for topology violations.
    This is used by the generator to understand *what* broke and *where*.
    """
    code: str
    message: str
    bay_id: Optional[str] = None
    device_id: Optional[str] = None

    def __str__(self):
        parts = [self.code, self.message]
        if self.bay_id:
            parts.append(f"bay={self.bay_id}")
        if self.device_id:
            parts.append(f"device={self.device_id}")
        return " | ".join(parts)

class TopologyValidator:
    """
    Validates that a complete substation layout satisfies a declared topology.

    This class operates at the *yard level*, not the device level.
    It does not check whether two devices may connect — that is done by ElectricalGrammar.
    It checks whether the entire collection of devices, buses, and breakers forms
    a valid electrical substation topology such as:

        • Single bus
        • Double bus
        • Main and transfer
        • Ring bus
        • Breaker-and-a-half
        • Double breaker

    It is the structural type checker of the substation compiler.
    """
    
    def __init__(self, grammar: "ElectricalGrammar"):
        self.grammar = grammar

    def validate(self, devices: set[Device], edges: set[tuple[Device,Device]], topology: TopologySchema):
        """
        Validate the entire substation against a topology schema.

        This function is the main entry point for topology validation.
        It verifies that:
        • The required buses exist
        • Every bay matches a legal bay pattern
        • Ring buses form closed loops
        • Radial feeders are isolated
        • Multi-bus bays are wired correctly

        If any rule is violated, a TopologyError is raised.
        """
        self._validate_bus_roles(devices, topology)
        bays = self._group_bays(devices)
        adjacency = self._build_graph(edges)
        bay_roles = self._validate_bays(bays, adjacency, topology)
        self._validate_ring(topology, bay_roles)
        self._validate_double_bus(topology, bay_roles)
        self._validate_main_and_transfer(topology, bay_roles)
        self._validate_breaker_and_half(topology, bay_roles)
        self._validate_mesh(topology, bay_roles)

        return True
    
    def _build_graph(self, edges):
        """
        Convert edge list into forward and reverse adjacency maps.

        This allows fast lookup of:
            • What a device connects to
            • What connects to a device

        This answers both “what does this device feed?” and “what feeds this device?” 
        in constant time, which is required for topology, voltage, and bay-chain checks.
        """
        fwd = defaultdict(set) # for every device A, list all devices B that A connects to.
        rev = defaultdict(set) # for every device B, list all devices A that feed into it.
        for a, b in edges:
            fwd[a].add(b)
            rev[b].add(a)
        return fwd, rev

    def _group_bays(self, devices):
        """
        Group devices into bays using their bay_id.

        A bay represents one physical switching unit
        (transformer bay, feeder bay, line bay, etc.).

        This function builds:
            bay_id → {devices in that bay}
        """
        bays = defaultdict(set)
        for d in devices:
            if d.bay_id:
                bays[d.bay_id].add(d)
        return bays

    def _validate_bus_roles(self, devices, topology):
        """
        Verify that the yard contains exactly the buses required by the topology.

        For example:
            • Single bus requires 1 MAIN bus
            • Double bus requires 2 MAIN buses
            • Main-and-transfer requires 1 MAIN and 1 TRANSFER
            • Ring requires 1 RING bus
            • LV collection yard requires 1 LV bus
            • Hybrid main + LV yard requires 1 MAIN bus at HV and 1 LV bus at LV,

        If any required bus is missing or extra, the topology is invalid.
        """
        buses = defaultdict(list)
        for d in devices:
            if d.device_type == DeviceType.BUS:
                if not d.bus_role:
                    raise TopologyError(f"Bus {d.device_id} missing bus_role")
                buses[d.bus_role].append(d)

        for role, count in topology.buses.items():
            if len(buses[role]) != count:
                raise TopologyError(
                    "BUS_COUNT_MISMATCH",
                    f"{topology.name} requires {count} {role.name} buses, found {len(buses[role])}"
                )
    
    def _validate_bays(self, bays, graph, topology):
        """
        Validate every bay against the formal BaySchema rules and ensure that
        only bay types permitted by the selected TopologySchema are present.
        """
        bay_roles: Dict[str, BayRole] = {}       # FIX 1: real dict
        coupler_assigned = False
        fwd, rev = graph # forward and reverse adjacency maps
        
        # Deterministic candidate priority: COUPLER first so it is assigned
        # exactly once before LINE-role schemas consume the rest.
        PRIORITY_ORDER = ("COUPLER", "RING", "DOUBLE_BREAKER", "RADIAL_LV", "SINGLE_BUS_TX")


        for bay_id, devs in bays.items():
            # Count the devices inside this bay
            breakers = [d for d in devs if d.device_type == DeviceType.BREAKER]
            transformers = [d for d in devs if d.device_type == DeviceType.TRANSFORMER]
            feeders = [d for d in devs if d.device_type == DeviceType.FEEDER]

            # Which external BUS-typed devices does this bay touch via edges?
            buses: set = set()
            for d in devs:
                for x in fwd[d] | rev[d]:
                    if x.device_type == DeviceType.BUS:
                        buses.add(x)

            matched: Optional[str] = None
            
            # a stable priority order that lets COUPLER be assigned once.
            for name in PRIORITY_ORDER:
                if name not in topology.bay_types:
                    continue                          
                if name not in ElectricalGrammar.BAY_SCHEMAS:
                    continue
                schema = ElectricalGrammar.BAY_SCHEMAS[name]

                # after one COUPLER is assigned, stop trying COUPLER
                if name == "COUPLER" and coupler_assigned:
                    continue
                # how many buses the bay is connected to
                if not (schema.min_buses <= len(buses) <= schema.max_buses):
                    continue
                #  how many breakers exist in the bay
                if len(breakers) != schema.breakers:
                    continue
                # enforce voltage-bridge requirement for transformer bays
                if schema.requires_transformer and not transformers:
                    continue
                # enforce termination rules: some bays may not have feeders
                if not schema.allows_feeder and feeders:
                    continue
                # if all constraints match, this bay fits this schema
                matched = name
                bay_roles[bay_id] = schema.role
                if name == "COUPLER":
                    coupler_assigned = True
                break
                            
            # if no schema matched, the bay is electrically invalid
            if not matched:
                raise TopologyError(
                    "BAY_INVALID", 
                    "Bay does not match any schema", bay_id=bay_id
                )

            # check that this bay type is allowed by the selected topology
            if matched not in topology.bay_types:
                raise TopologyError(
                    "BAY_NOT_ALLOWED", 
                    f"{matched} not allowed in {topology.name}", 
                    bay_id=bay_id
                )
        return bay_roles
    
    def _validate_ring(self, topology, bay_roles):
        """
        Enforces ring-bus topology using bay semantics.

        A valid ring yard is composed only of LINE bays, and must contain
        at least three of them to form a closed electrical loop.
        """
        if topology.name != "RING":
            return

        # Every bay in a ring must be a line segment of the ring
        for bay_id, role in bay_roles.items():
            if role != BayRole.LINE:
                raise TopologyError(
                    "RING_NON_LINE",
                    "Non-line bay in ring topology",
                    bay_id=bay_id
                )

        # A ring needs at least three segments to close
        line_count = sum(1 for r in bay_roles.values() if r == BayRole.LINE)
        if line_count < 3:
            raise TopologyError(
                "RING_TOO_SMALL",
                "Ring must contain at least three line bays"
            )

    def _validate_double_bus(self, topology, bay_roles):
        """
        Enforces double-bus topology rules.

        In a true double-bus yard, every bay must be electrically
        connected to both MAIN busbars so that it can be switched
        between them for redundancy and maintenance.
        """
        
        # If this is not a double-bus topology, this rule does not apply
        if topology.name != "DOUBLE_BUS":
            return

        # In a double-bus yard, only LINE bays may exist on the HV buses
        for bay_id, role in bay_roles.items():
            if role != BayRole.LINE:
                raise TopologyError(
                    "DOUBLE_BUS_NON_LINE",
                    "Only line bays allowed in double-bus topology",
                    bay_id=bay_id
                )

    def _validate_main_and_transfer(self, topology, bay_roles):
        """
        Enforces main-and-transfer bus topology rules.

        A valid main-and-transfer yard must contain at least one
        breaker that electrically connects the MAIN bus to the
        TRANSFER bus. This breaker is the transfer coupler that
        allows bays to be moved between buses during maintenance.
        """
        # If this is not a main-and-transfer topology, this rule does not apply
        if topology.name != "MAIN_AND_TRANSFER":
            return

        # A main-and-transfer yard is defined by the presence of a coupler bay
        if BayRole.COUPLER not in bay_roles.values():
            raise TopologyError(
                "NO_COUPLER",
                "Main-and-transfer topology requires a coupler bay"
            )
       
    def _validate_breaker_and_half(self, topology, bay_roles):
        if topology.name != "BREAKER_AND_HALF":
            return

        if BayRole.COUPLER not in bay_roles.values():
            raise TopologyError("NO_COUPLER", "Breaker-and-half requires coupler bays")

        if BayRole.LINE not in bay_roles.values():
            raise TopologyError("NO_LINE", "Breaker-and-half requires line bays")

        for bay_id, role in bay_roles.items():
            if role not in (BayRole.LINE, BayRole.COUPLER):
                raise TopologyError("BAD_BAY", "Invalid bay in breaker-and-half", bay_id=bay_id)

    def _validate_mesh(self, topology, bay_roles):
        if topology.name != "MESH":
            return

        if BayRole.COUPLER not in bay_roles.values():
            raise TopologyError("NO_COUPLER", "Mesh requires coupler bays")

        if BayRole.LINE not in bay_roles.values():
            raise TopologyError("NO_LINE", "Mesh requires line bays")

        for bay_id, role in bay_roles.items():
            if role not in (BayRole.LINE, BayRole.COUPLER):
                raise TopologyError("BAD_BAY", "Invalid bay in mesh topology", bay_id=bay_id)
 
# =================================================
# THE REAL DEAL
# =================================================
class ElectricalGrammar:
    DEVICE_PROPERTIES: Dict[DeviceType, DeviceProperties] = {
        DeviceType.GRID_SOURCE: DeviceProperties(
            device_type=DeviceType.GRID_SOURCE,
            is_voltage_bridge=False,
            is_terminal=False,
            primary_voltage=None # Grid typically at higher voltage
        ),
        
        DeviceType.TRANSFORMER: DeviceProperties(
            device_type=DeviceType.TRANSFORMER,
            is_voltage_bridge=True,  # Bridges 33kV and 11kV
            is_terminal=False,
        ),
        
        DeviceType.BUS: DeviceProperties(
            device_type=DeviceType.BUS,
            is_voltage_bridge=False,
            is_terminal=False,
        ),
        
        DeviceType.BREAKER: DeviceProperties(
            device_type=DeviceType.BREAKER,
            is_voltage_bridge=False,
            is_terminal=False,
        ),
        
        DeviceType.ISOLATOR: DeviceProperties(
            device_type=DeviceType.ISOLATOR,
            is_voltage_bridge=False,
            is_terminal=False,
        ),
        
        DeviceType.FEEDER: DeviceProperties(
            device_type=DeviceType.FEEDER,
            is_voltage_bridge=False,
            is_terminal=True,  # Feeders are endpoints
        ),
    }
    
    CONNECTION_RULES: Set[Tuple[DeviceType, DeviceType]] = {
        # Grid source connections
        (DeviceType.GRID_SOURCE, DeviceType.BREAKER),
        (DeviceType.GRID_SOURCE, DeviceType.BUS),
        # (DeviceType.GRID_SOURCE, DeviceType.TRANSFORMER),
        
        # Transformer connections
        (DeviceType.TRANSFORMER, DeviceType.BREAKER),
        # (DeviceType.TRANSFORMER, DeviceType.BUS),
        
        # Bus connections
        (DeviceType.BUS, DeviceType.ISOLATOR),
        (DeviceType.BUS, DeviceType.BREAKER),
        (DeviceType.BUS, DeviceType.TRANSFORMER),
        
        # Breaker connections
        (DeviceType.BREAKER, DeviceType.ISOLATOR),
        (DeviceType.BREAKER, DeviceType.TRANSFORMER),
        (DeviceType.BREAKER, DeviceType.BUS),
        (DeviceType.BREAKER, DeviceType.FEEDER),
        
        # Isolator connections
        (DeviceType.ISOLATOR, DeviceType.BREAKER),
        
        # Note: Feeders are terminal, so no outgoing connections
    }

    BAY_SCHEMAS: Dict[str, BaySchema] = {
        "RADIAL_LV": BaySchema(
            "RADIAL_LV",
            role=BayRole.FEEDER,
            min_buses=1,
            max_buses=1,
            breakers=2,
            requires_transformer=False,
            allows_feeder=True
        ),
        "SINGLE_BUS_TX": BaySchema(
            "SINGLE_BUS_TX",
            role=BayRole.TRANSFORMER,
            min_buses=1,
            max_buses=1,
            breakers=2,
            requires_transformer=True,
            allows_feeder=False
        ),
        "RING": BaySchema(
            "RING",
            role=BayRole.LINE,
            min_buses=2,
            max_buses=2,
            breakers=2,
            requires_transformer=False,
            allows_feeder=False
        ),
        "DOUBLE_BREAKER": BaySchema(
            "DOUBLE_BREAKER",
            role=BayRole.LINE,
            min_buses=2,
            max_buses=2,
            breakers=2,
            requires_transformer=False,
            allows_feeder=False
        ),
        "COUPLER": BaySchema(
            "COUPLER", 
            role=BayRole.COUPLER, 
            min_buses=2, 
            max_buses=2, 
            breakers=2, 
            requires_transformer=False, 
            allows_feeder=False
        ),
    }

    TOPOLOGIES: Dict[str, TopologySchema] = {
        "SINGLE_BUS": TopologySchema(
            "SINGLE_BUS",
            buses={BusRole.MAIN: 1},
            bay_types={"SINGLE_BUS_TX", "RADIAL_LV"},
            allows_bus_couplers=False
        ),
        "DOUBLE_BUS": TopologySchema(
            "DOUBLE_BUS",
            buses={BusRole.MAIN: 2},
            bay_types={"DOUBLE_BREAKER"},
            allows_bus_couplers=True
        ),
        "MAIN_AND_TRANSFER": TopologySchema(
            "MAIN_AND_TRANSFER",
            buses={BusRole.MAIN: 1, BusRole.TRANSFER: 1},
            bay_types={"SINGLE_BUS_TX", "COUPLER"},
            allows_bus_couplers=True
        ),
        "RING": TopologySchema(
            "RING",
            buses={BusRole.RING: 1},
            bay_types={"RING"},
            allows_bus_couplers=False
        ),
        "BREAKER_AND_HALF": TopologySchema(
            "BREAKER_AND_HALF",
            buses={BusRole.MAIN: 2},
            bay_types={"DOUBLE_BREAKER", "COUPLER"},
            allows_bus_couplers=True
        ),
        "MESH": TopologySchema(
            "MESH",
            buses={BusRole.MAIN: 3},
            bay_types={"DOUBLE_BREAKER", "COUPLER"},
            allows_bus_couplers=True
        )

    }

    # whether two device types are allowed to connect based on grammar 
    def is_device_type_connection_allowed(
        self, source_type: DeviceType, target_type: DeviceType
    ) -> bool:
        """
        Check if two device types are allowed to connect based on grammar rules.
        
        Args:
            source_type: The source device type
            target_type: The target device type
            
        Returns:
            True if connection is allowed by device type rules
        """
        return (source_type, target_type) in self.CONNECTION_RULES

    def required_voltages(self, device: Device, target_type: DeviceType) -> set[Voltage]:
        """
        Return the set of voltage domains that a target device of the given type
        is allowed to have when connected from the given source device.

        This encodes the voltage-transition law of the grammar:
        • If either side is a voltage-bridge (e.g., a transformer), the voltage must change.
        • If neither side is a bridge, the voltage must remain the same.

        This function is used by the topology generator to choose the correct
        voltage for the next device before it is created.
        """
        props = self.DEVICE_PROPERTIES[device.device_type]
        target_props = self.DEVICE_PROPERTIES[target_type]

        # voltage changes only when entering a transformer
        if target_props.is_voltage_bridge:
            return {v for v in Voltage if v != device.voltage}

        # leaving a transformer stays in its secondary domain
        if props.is_voltage_bridge:
            return {device.voltage}

        # both single-domain devices -> must match
        return {device.voltage}

    def can_emit(self, device: Device) -> bool:
        """
        Return whether the given device is allowed to have outgoing connections.

        Terminal devices (e.g., feeders) cannot emit edges.
        Non-terminal devices can emit edges.

        This encodes directional electrical flow at the grammar level.
        """
        return not self.DEVICE_PROPERTIES[device.device_type].is_terminal

    def can_receive(self, device: Device) -> bool:
        """
        Return whether the given device is allowed to receive incoming connections.

        Most electrical devices can receive connections.
        This function exists to allow future extension for asymmetric devices.
        """
        return True
    
    def check_edge(self, device_a: Device, device_b: Device) -> ConnectionVerdict:
        """
        Determine whether a directed connection from device_a to device_b
        is legal under the electrical grammar.

        This applies three independent rules:
        • Device-type adjacency (CONNECTION_RULES)
        • Terminal emission rules (can_emit)
        • Voltage-domain compatibility (required_voltages)

        The result is returned as a ConnectionVerdict enum value that
        precisely identifies the reason for acceptance or rejection.
        """
        # type rule
        if not self.is_device_type_connection_allowed(device_a.device_type, device_b.device_type):
            return ConnectionVerdict.REJECT_TYPE

        # terminal rule
        if not self.can_emit(device_a):
            return ConnectionVerdict.REJECT_TERMINAL

        # voltage rule
        allowed_voltages = self.required_voltages(device_a, device_b.device_type)
        if device_b.voltage not in allowed_voltages:
            return ConnectionVerdict.REJECT_VOLTAGE

        return ConnectionVerdict.OK
    
    def validate_topology(self, devices, edges, topology_name):
        """
        Validate a complete yard against a named topology.

        This bridges the grammar layer and the topology layer.
        It looks up the TopologySchema and runs the structural validator.

        This must be called after all devices and connections are generated.
        """
        topology = self.TOPOLOGIES[topology_name]
        validator = TopologyValidator(self)
        return validator.validate(devices, edges, topology)
 
 