# Give each dataset an independent object closure

Each standalone dataset serializes the complete TSD object closure reachable
from its layer subtree, and loading it creates dataset-owned objects. Object
identity is never shared across dataset files: equivalent materials, samplers,
arrays, fields, or other scene objects may be duplicated, but editing one
dataset cannot mutate another through an implicit shared object.
