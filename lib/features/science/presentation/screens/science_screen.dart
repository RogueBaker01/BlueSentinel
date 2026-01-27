import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';
import '../widgets/density_heatmap.dart';
import '../widgets/population_chart.dart';

class ScienceScreen extends StatelessWidget {
  const ScienceScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text('Science & Data'),
        backgroundColor: Colors.white,
        centerTitle: false,
        elevation: 0,
        titleTextStyle: const TextStyle(
          color: AppTheme.deepOceanBlue,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share, color: AppTheme.deepOceanBlue),
            onPressed: () {},
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Global Heatmap',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 12),
          const DensityHeatmap(),
          const SizedBox(height: 24),
          const Text(
            'Population Statistics',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 12),
          const PopulationChart(),
          const SizedBox(height: 24),
          _buildInfoCard(
            context,
            "Predicción de Migración",
            "Basado en los cambios de temperatura del agua (+0.5°C), se espera que el 30% de la población de tiburones tigre migre hacia el norte en los próximos 15 días.",
            Icons.query_stats,
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(
    BuildContext context,
    String title,
    String content,
    IconData icon,
  ) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.accentTurquoise.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.accentTurquoise.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: AppTheme.deepOceanBlue),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppTheme.deepOceanBlue,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            content,
            style: const TextStyle(
              color: AppTheme.deepOceanBlue,
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}
