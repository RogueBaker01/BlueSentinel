import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../../../../core/theme/app_theme.dart';

class DensityHeatmap extends StatelessWidget {
  const DensityHeatmap({super.key});

  @override
  Widget build(BuildContext context) {
    // Datos simulados de densidad (Puntos calientes)
    final points = [
      const LatLng(25.0, -90.0), // Golfo
      const LatLng(-2.0, -91.0), // Galapagos
      const LatLng(10.0, -85.0), // Costa Rica
      const LatLng(18.0, -65.0), // Caribe
      const LatLng(-5.0, -35.0), // Brasil costa
    ];

    return Container(
      height: 300,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Stack(
          children: [
            FlutterMap(
              options: const MapOptions(
                initialCenter: LatLng(10.0, -80.0),
                initialZoom: 3.0,
                interactionOptions: InteractionOptions(
                  flags:
                      InteractiveFlag.none, // Mapa estático para visualización
                ),
              ),
              children: [
                TileLayer(
                  urlTemplate:
                      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png', // Mapa oscuro para contraste
                  userAgentPackageName: 'com.bluesentinel.app',
                ),
                CircleLayer(
                  circles: points
                      .map(
                        (p) => CircleMarker(
                          point: p,
                          color: AppTheme.alertRed.withValues(alpha: 0.3),
                          borderStrokeWidth: 0,
                          radius:
                              400000, // Radio grande para efecto difuso (en metros aprox visualmente)
                          useRadiusInMeter: true,
                        ),
                      )
                      .toList(),
                ),
                CircleLayer(
                  circles: points
                      .map(
                        (p) => CircleMarker(
                          point: p,
                          color: AppTheme.alertRed.withValues(alpha: 0.5),
                          borderStrokeWidth: 0,
                          radius: 200000,
                          useRadiusInMeter: true,
                        ),
                      )
                      .toList(),
                ),
              ],
            ),
            Positioned(
              top: 16,
              left: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.7),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  children: [
                    Icon(
                      Icons.local_fire_department,
                      color: AppTheme.alertRed,
                      size: 16,
                    ),
                    SizedBox(width: 8),
                    Text(
                      'Zonas de Alta Densidad',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
