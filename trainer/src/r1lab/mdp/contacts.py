"""Exact union of disjoint body targets without broadening contact filters.

mjlab's secondary ContactMatch accepts one body, not a scene-global regex.
Register native sensors for each target, then sum world net forces per primary
before inherited history and airtime tracking consume the data.
"""
from dataclasses import dataclass, fields

from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sensor.contact_sensor import _ContactSlot


@dataclass
class UnionContactSensorCfg(ContactSensorCfg):
  targets: tuple[str, ...] = ("terrain", "adversary_scene/scene")

  def build(self):
    return UnionContactSensor(self)


class UnionContactSensor(ContactSensor):
  def __init__(self, cfg):
    if cfg.reduce != "netforce" or cfg.num_slots != 1:
      raise ValueError("Union contacts require netforce and one slot per primary")
    if not cfg.targets or len(set(cfg.targets)) != len(cfg.targets):
      raise ValueError("Union targets must be distinct exact body names")
    if not set(cfg.fields) <= {"found", "force", "torque"}:
      raise ValueError("Only additive found/force/torque fields are supported")
    if cfg.secondary is None or cfg.secondary.mode != "body" or cfg.secondary.entity:
      raise ValueError("Union secondary must be a scene-global literal body match")
    super().__init__(cfg)

  def edit_spec(self, scene_spec, entities):
    self._slots.clear()
    for primary in self._resolve_primary_names(entities, self.cfg.primary):
      for field in self.cfg.fields:
        for index, target in enumerate(self.cfg.targets):
          name = f"{self.cfg.name}_{primary}_{field}_target{index}"
          # cfg.secondary supplies body mode and no prefix; target is literal.
          self._add_contact_sensor_to_spec(scene_spec, name, primary, target, field)
          self._slots.append(_ContactSlot(primary_name=primary, field_name=field,
                                          sensor_name=name))

  def _extract_sensor_data(self):
    data = super()._extract_sensor_data()
    primary_count, target_count = len(self.primary_names), len(self.cfg.targets)
    for field in self.cfg.fields:
      value = getattr(data, field)
      value = value.reshape(value.shape[0], primary_count, target_count, *value.shape[2:])
      setattr(data, field, value.sum(dim=2))
    return data


def terrain_union(sensor):
  """Preserve primary order, fields, history and airtime from an existing cfg."""
  values = {field.name: getattr(sensor, field.name) for field in fields(ContactSensorCfg)}
  values.update(secondary=ContactMatch(mode="body", pattern="terrain"),
                secondary_policy="error", reduce="netforce", num_slots=1)
  return UnionContactSensorCfg(**values)
