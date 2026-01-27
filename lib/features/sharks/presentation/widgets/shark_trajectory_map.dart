import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../../../../core/theme/app_theme.dart';
import '../../domain/models/shark.dart';

class SharkTrajectoryMap extends StatelessWidget {
  final Shark shark;

  const SharkTrajectoryMap({super.key, required this.shark});

  @override
  Widget build(BuildContext context) {
    // Generar path simulado (última posición conocida -> 5 puntos anteriores)
    final points = [
      LatLng(shark.lastLatitude, shark.lastLongitude),
      LatLng(shark.lastLatitude - 0.5, shark.lastLongitude + 0.2),
      LatLng(shark.lastLatitude - 1.2, shark.lastLongitude - 0.5),
      LatLng(shark.lastLatitude - 2.0, shark.lastLongitude + 0.8),
      LatLng(shark.lastLatitude - 3.5, shark.lastLongitude - 1.0),
    ];

    return FlutterMap(
      options: MapOptions(initialCenter: points.first, initialZoom: 6.0),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.bluesentinel.app',
        ),
        PolylineLayer(
          polylines: [
            Polyline(
              points: points,
              strokeWidth: 4.0,
              color: AppTheme.accentTurquoise,
            ),
          ],
        ),
        MarkerLayer(
          markers: [
            // Marcador actual (Posición real)
            Marker(
              point: points.first,
              width: 80,
              height: 80,
              child: const Icon(
                Icons.location_on,
                color: AppTheme.alertRed,
                size: 40,
              ),
            ),
            // Inicio de trayectoria
            Marker(
              point: points.last,
              width: 40,
              height: 40,
              child: Container(
                decoration: const BoxDecoration(
                  color: AppTheme.deepOceanBlue,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.flag, color: Colors.white, size: 20),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
