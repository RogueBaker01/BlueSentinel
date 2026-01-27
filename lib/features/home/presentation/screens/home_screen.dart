import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';
import 'package:go_router/go_router.dart';
import '../widgets/home_map.dart';
import '../widgets/map_active_filters.dart';
import '../widgets/recent_activity_card.dart';
import '../widgets/weather_widget.dart';
import '../../../sharks/presentation/screens/sharks_screen.dart';
import '../../../science/presentation/screens/science_screen.dart';
import '../../../education/presentation/screens/education_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  List<Widget> get _widgetOptions => <Widget>[
    const HomeMap(),
    const SharksScreen(),
    const ScienceScreen(),
    const EducationScreen(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // 1. Capa Base: Mapa
          Positioned.fill(child: _widgetOptions.elementAt(_selectedIndex)),

          // 2. Capa Superior: Filtros y UI (Solo visible en tab de Mapa)
          if (_selectedIndex == 0)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Fila de Filtros
                      const MapActiveFilters(),

                      const SizedBox(height: 16),

                      // Aquí irán futuras notificaciones
                    ],
                  ),
                ),
              ),
            ),

          // Widget de Clima (Top Right)
          if (_selectedIndex == 0)
            const Positioned(
              top: 0,
              right: 16,
              child: SafeArea(
                child: Padding(
                  padding: EdgeInsets.only(top: 16.0),
                  child: WeatherWidget(),
                ),
              ),
            ),

          // 2.5 Botón de Perfil (Top Left - debajo de filtros o al lado)
          // Lo pondremos arriba a la izquierda pero con margen superior para no chocar con filtros si están ahí,
          // O mejor: Reemplazamos la lógica de posición para organizar mejor.
          // Vamos a poner el botón de perfil flotante abajo a la izquierda, estilo "Google Maps Layers" pero para Settings.
          if (_selectedIndex == 0)
            Positioned(
              left: 16,
              bottom: 100, // Encima del RecentActivityCard que es un sheet
              child: FloatingActionButton(
                heroTag: 'settings_btn',
                backgroundColor: Colors.white,
                mini: true,
                child: const Icon(
                  Icons.settings,
                  color: AppTheme.deepOceanBlue,
                ),
                onPressed: () {
                  context.push('/settings');
                },
              ),
            ),

          // 3. Capa Inferior: Tarjeta de Actividad (Bottom Sheet)
          if (_selectedIndex == 0) const RecentActivityCard(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: _onItemTapped,
        backgroundColor: Colors.white,
        indicatorColor: AppTheme.surfaceBlue,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map, color: AppTheme.oceanBlue),
            label: 'Mapa',
          ),
          NavigationDestination(
            icon: Icon(Icons.radar_outlined),
            selectedIcon: Icon(Icons.radar, color: AppTheme.oceanBlue),
            label: 'Tiburones',
          ),
          NavigationDestination(
            icon: Icon(Icons.analytics_outlined),
            selectedIcon: Icon(Icons.analytics, color: AppTheme.oceanBlue),
            label: 'Datos',
          ),
          NavigationDestination(
            icon: Icon(Icons.school_outlined),
            selectedIcon: Icon(Icons.school, color: AppTheme.oceanBlue),
            label: 'Educación',
          ),
        ],
      ),
    );
  }
}
