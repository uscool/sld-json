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
from typing import Optional

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

class BusRole(Enum):
    MAIN = auto()
    TRANSFER = auto()
    RING = auto()
    LV = auto()

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