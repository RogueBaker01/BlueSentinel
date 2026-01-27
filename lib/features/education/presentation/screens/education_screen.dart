import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';
import '../widgets/news_card.dart';

class EducationScreen extends StatelessWidget {
  const EducationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50], // Fondo limpio
      appBar: AppBar(
        title: const Text('Ocean News & Edu'),
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
            icon: const Icon(
              Icons.bookmark_border,
              color: AppTheme.deepOceanBlue,
            ),
            onPressed: () {},
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildHeroSection(),
          const SizedBox(height: 24),
          const Text(
            'Latest Articles',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppTheme.deepOceanBlue,
            ),
          ),
          const SizedBox(height: 16),
          NewsCard(
            title: "Nuevo Santuario Marino Protegido en el Pacífico",
            category: "Conservación",
            imageUrl:
                "https://images.unsplash.com/photo-1682687220742-aba13b6e50ba?auto=format&fit=crop&q=80",
            timeAgo: "Hace 2 horas",
            onTap: () => context.push(
              '/article',
              extra: {
                'title': "Nuevo Santuario Marino Protegido en el Pacífico",
                'category': "Conservación",
                'image':
                    "https://images.unsplash.com/photo-1682687220742-aba13b6e50ba?auto=format&fit=crop&q=80",
                'time': "5 min lectura",
                'content':
                    "Una nueva área marina protegida ha sido establecida en el Océano Pacífico, cubriendo más de 500,000 kilómetros cuadrados. Esta iniciativa busca proteger las rutas migratorias de especies en peligro como el tiburón martillo y diversas especies de tortugas marinas.\n\nEl acuerdo internacional fue firmado por 4 naciones costeras, comprometiéndose a vigilancia satelital constante y patrullaje conjunto para evitar la pesca ilegal.",
              },
            ),
          ),
          NewsCard(
            title: "¿Por qué los tiburones son vitales para el ecosistema?",
            category: "Educación",
            imageUrl:
                "https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80",
            timeAgo: "Hace 1 día",
            onTap: () => context.push(
              '/article',
              extra: {
                'title':
                    "¿Por qué los tiburones son vitales para el ecosistema?",
                'category': "Educación",
                'image':
                    "https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80",
                'content':
                    "Como depredadores tope, los tiburones juegan un papel crucial en el mantenimiento de la salud de los océanos. Regulan las poblaciones de otras especies marinas, eliminan a los individuos enfermos o débiles, y aseguran la biodiversidad de los arrecifes de coral.\n\nSin ellos, el equilibrio de la cadena alimenticia colapsaría, afectando incluso la producción de oxígeno del fitoplancton.",
              },
            ),
          ),
          NewsCard(
            title: "Tecnología Satelital: El futuro del rastreo",
            category: "Tecnología",
            imageUrl:
                "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80",
            timeAgo: "Hace 3 días",
            onTap: () => context.push(
              '/article',
              extra: {
                'title': "Tecnología Satelital: El futuro del rastreo",
                'category': "Tecnología",
                'image':
                    "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80",
                'content':
                    "Los nuevos tags satelitales permiten un seguimiento en tiempo real con una precisión sin precedentes. Estos dispositivos, más pequeños y duraderos, transmiten datos de profundidad, temperatura y localización cada vez que el tiburón sale a la superficie.",
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeroSection() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [AppTheme.deepOceanBlue, AppTheme.oceanBlue],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: AppTheme.oceanBlue.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Tip del Día",
                  style: TextStyle(
                    color: AppTheme.accentTurquoise,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  "Evita plásticos de un solo uso. El 70% terminan en el océano.",
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.lightbulb, color: Colors.white, size: 32),
          ),
        ],
      ),
    );
  }
}
