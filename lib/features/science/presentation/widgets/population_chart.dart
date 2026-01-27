import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';

class PopulationChart extends StatelessWidget {
  const PopulationChart({super.key});

  @override
  Widget build(BuildContext context) {
    // Datos simulados: Especie -> Cantidad
    final data = {
      'White': 45,
      'Hammer': 32,
      'Whale': 18,
      'Tiger': 24,
      'Blue': 12,
    };
    final maxVal = 50;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Distribución de Especies',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppTheme.deepOceanBlue,
            ),
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: data.entries.map((entry) {
              final heightFactor = entry.value / maxVal;
              return Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Text(
                    '${entry.value}',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      color: AppTheme.oceanBlue,
                    ),
                  ),
                  const SizedBox(height: 8),
                  TweenAnimationBuilder<double>(
                    tween: Tween(begin: 0, end: heightFactor),
                    duration: const Duration(seconds: 1),
                    curve: Curves.bounceOut,
                    builder: (context, value, child) {
                      return Container(
                        width: 40,
                        height: 150 * value,
                        decoration: BoxDecoration(
                          color: AppTheme.oceanBlue.withValues(alpha: 0.8),
                          borderRadius: BorderRadius.circular(8),
                          gradient: LinearGradient(
                            begin: Alignment.bottomCenter,
                            end: Alignment.topCenter,
                            colors: [
                              AppTheme.deepOceanBlue,
                              AppTheme.oceanBlue,
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 8),
                  Text(
                    entry.key,
                    style: TextStyle(color: Colors.grey[600], fontSize: 12),
                  ),
                ],
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}
