import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class HomeMap extends StatelessWidget {
  const HomeMap({super.key});

  @override
  Widget build(BuildContext context) {
    return FlutterMap(
      options: MapOptions(
        initialCenter: const LatLng(
          -1.2833,
          -90.5333,
        ), // Islas Galápagos (Punto caliente de tiburones)
        initialZoom: 6.0,
        minZoom: 3.0,
        maxZoom: 18.0,
        interactionOptions: const InteractionOptions(
          flags:
              InteractiveFlag.all &
              ~InteractiveFlag
                  .rotate, // Desactivar rotación para mantener norte arriba
        ),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.bluesentinel.app',
          // Opcional: Usar un estilo más "oceánico" si tuviéramos un proveedor personalizado,
          // por ahora OSM estándar.
        ),
        // Aquí irán las capas de marcadores (tiburones) y heatmaps
        RichAttributionWidget(
          attributions: [
            TextSourceAttribution('OpenStreetMap contributors', onTap: () {}),
          ],
        ),
      ],
    );
  }
}
