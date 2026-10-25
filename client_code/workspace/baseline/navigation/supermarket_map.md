# Supermarket Sorting Map (from retail_competition.xml)

World frame: +X = east (right), +Y = north (up). Origin at floor center.
Floor bounds: x ∈ [-2.5, 2.5], y ∈ [-3.75, 3.75].

```
   y=3.75 ┌──────────────────────────────────────────────────┐
          │            SHELF ROW  (y = 3.323)                │
   y=3.32 │   [A]      [B]      [C]      [D]      [E]        │
          │  x=-1.74  -0.85    0.04     0.92     1.81        │
          │  (kele at aruco markers, front y≈3.243)          │
          │  ──────────────────────────────────────────────   │
   y=3.25 │  ██ picking_zone north yellow line (y=3.25) ██   │
          │  ████  SHELF APPROACH LANE  ████                 │
   y=2.48 │  ██ picking_zone mid  (approach x≈0.85) ██       │
          │  ██ picking_zone south yellow line (y=1.70) ██   │
   y=1.70 │  ──────────────────────────────────────────────   │
          │                                                   │
          │            OPEN AREA (no boxes)                   │
          │                                                   │
          │             ┌─ obstacle corridor ─┐               │
   y=1.70 │             │  ██ box5 (west) ██ │               │
   y=1.34 │             │  ████ ████  ████   │  east wall     │
          │             │  ██ box4 ██        │  x≈0.53        │
   y~0    │             │  ██ box3 ██        │               │
          │             │  ██ box2 ██        │               │
   y=-1.0 │             │  ██ box1 ██        │               │
          │             └────────────────────┘               │
          │  corridor: x[-2.47,0.50] y[-3.72,1.70]           │
          │  (boxes randomized, 5 slalom bands)              │
          │                                                  │
   y=-2.8 │              delivery zone  [yellow line]         │
   y=-3.4 │              ██ TABLE (-1.94,-3.41) ██           │
          │                                                  │
   y=-3.2 │            START ZONE (1.92,-3.17)  ◉ robot      │
          │                                                  │
   y=-3.75└──────────────────────────────────────────────────┘
          x=-2.5                                         x=2.5
```

## Key coordinates

| Element | World pos | Notes |
|---|---|---|
| Start zone | (1.92, -3.17) | bottom-right, robot spawns here facing north (+Y) |
| Picking zone | y line 1.70..3.25, mid y=2.475 | yellow lines; shelf approach lane at x≈0.85 |
| Shelf A..E | y=3.323; x = -1.735, -0.85, 0.035, 0.92, 1.805 | products/kele at front y≈3.243, z≈0.55/0.92/1.22 |
| Obstacle corridor | x[-2.47, 0.50], y[-3.72, 1.70] | boxes randomized (0.60×0.40 m), 5 slalom bands, east wall x≈0.53 |
| Delivery table | (-1.94, -3.41) | bottom-left, top z≈0.77, enclosed by yellow line |
| Delivery target | (-1.94, -3.41, 0.807) | placement spot |

## Route logic (matches the code)

1. **NAV_SHELF**: from start (1.92,-3.17) drive north up the OPEN east side
   (no boxes) to waypoint (1.92, 2.475), then west along the yellow picking
   zone to the approach lane (0.852, 2.475). Grasp kele from shelf.
2. **NAV_TABLE**: turn around, drive back south THROUGH the obstacle corridor
   (boxes here), exiting to the delivery table at (-1.88, -2.80) and place on
   the yellow-enclosed table.

## Obstacle avoidance requirement

The boxes live ONLY inside the corridor (x[-2.47,0.50], y[-3.72,1.70]).
The path from start→shelf is box-free; the return trip south through the
corridor is where lidar DWA avoidance is needed to dodge the slalom boxes.
