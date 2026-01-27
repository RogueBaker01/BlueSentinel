import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../../core/theme/app_theme.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<Map<String, String>> _pages = [
    {
      'title': 'Bienvenido a BlueSentinel',
      'body':
          'La plataforma avanzada para el monitoreo y conservación de tiburones en tiempo real.',
      'icon': 'ocean',
    },
    {
      'title': 'Rastreo Inteligente',
      'body':
          'Sigue la trayectoria de especies protegidas mediante nuestro sistema de tags satelitales.',
      'icon': 'radar',
    },
    {
      'title': 'Ciencia Predictiva',
      'body':
          'Utilizamos Machine Learning para identificar zonas de alto riesgo y patrones de comportamiento.',
      'icon': 'analytics',
    },
  ];

  void _nextPage() {
    if (_currentPage < _pages.length - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    } else {
      context.go('/home');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.deepOceanBlue,
      body: Stack(
        children: [
          // Background Gradient
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [AppTheme.deepOceanBlue, Color(0xFF001020)],
              ),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                Expanded(
                  child: PageView.builder(
                    controller: _pageController,
                    onPageChanged: (index) {
                      setState(() {
                        _currentPage = index;
                      });
                    },
                    itemCount: _pages.length,
                    itemBuilder: (context, index) {
                      return Padding(
                        padding: const EdgeInsets.all(40.0),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              padding: const EdgeInsets.all(30),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: AppTheme.oceanBlue.withValues(
                                  alpha: 0.3,
                                ),
                                border: Border.all(
                                  color: AppTheme.accentTurquoise,
                                  width: 2,
                                ),
                              ),
                              child: Icon(
                                index == 0
                                    ? Icons.water
                                    : index == 1
                                    ? Icons.radar
                                    : Icons.analytics,
                                size: 80,
                                color: AppTheme.conservationGreen,
                              ),
                            ).animate().scale(
                              duration: 500.ms,
                              curve: Curves.easeOutBack,
                            ),

                            const SizedBox(height: 48),

                            Text(
                                  _pages[index]['title']!,
                                  textAlign: TextAlign.center,
                                  style: GoogleFonts.inter(
                                    fontSize: 28,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                  ),
                                )
                                .animate()
                                .fadeIn(delay: 200.ms)
                                .slideY(begin: 0.1, end: 0),

                            const SizedBox(height: 16),

                            Text(
                              _pages[index]['body']!,
                              textAlign: TextAlign.center,
                              style: GoogleFonts.lato(
                                fontSize: 16,
                                color: Colors.white70,
                                height: 1.5,
                              ),
                            ).animate().fadeIn(delay: 400.ms),
                          ],
                        ),
                      );
                    },
                  ),
                ),

                // Dots Indicator
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(
                    _pages.length,
                    (index) => AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      height: 8,
                      width: _currentPage == index ? 24 : 8,
                      decoration: BoxDecoration(
                        color: _currentPage == index
                            ? AppTheme.accentTurquoise
                            : Colors.white24,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // Action Button
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40,
                    vertical: 20,
                  ),
                  child: SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      onPressed: _nextPage,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.accentTurquoise,
                        foregroundColor: AppTheme.deepOceanBlue,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        elevation: 0,
                      ),
                      child: Text(
                        _currentPage == _pages.length - 1
                            ? 'Explorar Océano'
                            : 'Siguiente',
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ).animate().fadeIn(delay: 600.ms),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
