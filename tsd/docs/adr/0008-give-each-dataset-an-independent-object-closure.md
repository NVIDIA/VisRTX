# Give each dataset an independent object closure

Each Dataset Archive serializes the complete TSD object closure reachable from
its layer subtree, and loading it creates dataset-owned objects. Object identity
is never shared across Dataset Archives: equivalent materials, samplers, arrays,
fields, or other scene objects may be duplicated, but editing one dataset cannot
mutate another through an implicit shared object.
