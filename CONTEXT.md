# Railing removal

The input is a plant-candidate point cloud plus multi-view semantic evidence.
A confirmed seed is a point where railing evidence exceeds paired plant
evidence. A rail line is a long, narrow rigid structure supported by confirmed
low-green, low-saturation seeds. Completion fills unobserved points along an
accepted rail line. Strong plant evidence protects intersecting plant points.

The output is a rejection mask and an audit report. It never edits the source
cloud.
