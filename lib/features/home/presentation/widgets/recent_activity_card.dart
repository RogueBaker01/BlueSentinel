import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';

class RecentActivityCard extends StatelessWidget {
  const RecentActivityCard({super.key});

  @override
  Widget build(BuildContext context) {
    // Datos mockeados para simular actividad
    final activities = [
      {
        'species': 'Whale Shark',
        'location': 'near Holbox',
        'time': '2h ago',
        'image':
            'https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80',
        'color': AppTheme.oceanBlue,
      },
      {
        'species': 'Great White',
        'location': 'near Guadalupe Is.',
        'time': '5h ago',
        'image':
            'https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80', // Placeholder, idealmente variada
        'color': AppTheme.alertRed,
      },
      {
        'species': 'Hammerhead',
        'location': 'near Cocos Is.',
        'time': '1d ago',
        'image':
            'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80',
        'color': AppTheme.conservationGreen,
      },
    ];

    return DraggableScrollableSheet(
      initialChildSize: 0.25,
      minChildSize: 0.15,
      maxChildSize: 0.6,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.1),
                blurRadius: 10,
                offset: const Offset(0, -5),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Handle bar pill
              Center(
                child: Container(
                  margin: const EdgeInsets.only(top: 12, bottom: 8),
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20.0,
                  vertical: 8.0,
                ),
                child: Text(
                  'Recent Activity',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: AppTheme.deepOceanBlue,
                  ),
                ),
              ),

              Expanded(
                child: ListView.separated(
                  controller: scrollController,
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: activities.length,
                  separatorBuilder: (_, _) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final item = activities[index];
                    return ListTile(
                      leading: Container(
                        padding: const EdgeInsets.all(2), // Borde
                        decoration: BoxDecoration(
                          color: item['color'] as Color,
                          shape: BoxShape.circle,
                        ),
                        child: CircleAvatar(
                          radius: 18,
                          backgroundImage: NetworkImage(
                            item['image'] as String,
                          ),
                          backgroundColor: Colors.grey[200],
                        ),
                      ),
                      title: Text(
                        item['species'] as String,
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      subtitle: Text(item['location'] as String),
                      trailing: Text(
                        item['time'] as String,
                        style: const TextStyle(
                          color: Colors.grey,
                          fontSize: 12,
                        ),
                      ),
                      onTap: () {
                        // Navegar al detalle (Fase 3)
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
