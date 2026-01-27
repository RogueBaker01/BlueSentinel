import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';

class MapActiveFilters extends StatefulWidget {
  const MapActiveFilters({super.key});

  @override
  State<MapActiveFilters> createState() => _MapActiveFiltersState();
}

class _MapActiveFiltersState extends State<MapActiveFilters> {
  // Estado local para los filtros seleccionados
  // En el futuro, esto se movería a un Riverpod provider
  final Set<String> _selectedFilters = {'Species'};

  final List<String> _filters = ['Species', 'Environment', 'Activity'];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 40,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: _filters.length,
        separatorBuilder: (context, index) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final filter = _filters[index];
          final isSelected = _selectedFilters.contains(filter);

          return FilterChip(
            label: Text(filter),
            selected: isSelected,
            onSelected: (bool selected) {
              setState(() {
                if (selected) {
                  _selectedFilters.add(filter);
                } else {
                  _selectedFilters.remove(filter);
                }
              });
            },
            backgroundColor: Colors.white,
            selectedColor: AppTheme.oceanBlue,
            checkmarkColor: Colors.white,
            labelStyle: TextStyle(
              color: isSelected ? Colors.white : AppTheme.deepOceanBlue,
              fontWeight: FontWeight.w600,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
              side: BorderSide(
                color: isSelected
                    ? Colors.transparent
                    : Colors.grey.withValues(alpha: 0.3),
              ),
            ),
            elevation: 2,
            shadowColor: Colors.black.withValues(alpha: 0.3),
            showCheckmark: false, // Estilo más limpio tipo "botón"
          );
        },
      ),
    );
  }
}
