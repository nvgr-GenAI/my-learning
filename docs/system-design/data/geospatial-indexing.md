# Geospatial Indexing

**Efficiently query location-based data** | 📍 Geohash | 🗺️ Quadtree | 🌐 R-tree

---

## Overview

*Content coming soon.*

Location-based queries like "find restaurants near me" or "match riders with drivers" require specialized indexing. Standard B-tree indexes can't efficiently query two-dimensional spatial data.

---

## Topics to Cover

- **Geohash** — Encode lat/lng into string, prefix-based proximity, used by Redis GEO, Elasticsearch
- **Quadtree** — Recursive 2D space subdivision, adaptive density, used by Uber H3
- **R-tree** — Bounding rectangle hierarchy, range queries, used by PostGIS, MongoDB
- **S2 Geometry** — Google's hierarchical cell system, used by Google Maps, Foursquare
- **H3** — Uber's hexagonal hierarchical index, uniform area cells
- **Comparison** — Geohash vs Quadtree vs R-tree trade-offs (build time, query time, edge cases)
- **Proximity Search** — k-nearest neighbors, radius queries, bounding box queries
- **Sharding Geospatial Data** — Location-based partitioning, hotspot handling (dense cities)
- **Real-world Examples** — Uber (H3), Google Maps (S2), Yelp (Elasticsearch geohash), Pokemon Go

---

## Interview Relevance

- Critical for: Uber, Yelp, Google Maps, Tinder, food delivery designs
- Key question: "How do you find nearby drivers efficiently?"
- Geohash is the most commonly expected answer — simple, Redis-supported, shardable

---

## Related Topics

- [Database Indexing](databases/indexing.md)
- [Database Types](databases/database-types.md)
- [Sharding](databases/sharding.md)
