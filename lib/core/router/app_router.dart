import 'package:go_router/go_router.dart';
import '../../features/onboarding/presentation/screens/splash_screen.dart';
import '../../features/onboarding/presentation/screens/onboarding_screen.dart';
import '../../features/home/presentation/screens/home_screen.dart';
import '../../features/sharks/presentation/screens/shark_details_screen.dart';
import '../../features/settings/presentation/screens/settings_screen.dart';
import '../../features/education/presentation/screens/article_detail_screen.dart';

final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(path: '/', builder: (context, state) => const SplashScreen()),
    GoRoute(
      path: '/onboarding',
      builder: (context, state) => const OnboardingScreen(),
    ),
    GoRoute(path: '/home', builder: (context, state) => const HomeScreen()),
    GoRoute(
      path: '/shark-details',
      builder: (context, state) {
        final shark =
            state.extra
                as dynamic; // Cast dinámico temporal para evitar problemas de import circular si los hubiera, idealmente importar Shark
        return SharkDetailsScreen(shark: shark);
      },
    ),
    GoRoute(
      path: '/settings',
      builder: (context, state) => const SettingsScreen(),
    ),
    GoRoute(
      path: '/article',
      builder: (context, state) {
        // Cast seguro recuperando el mapa
        final article = state.extra as Map<String, dynamic>;
        return ArticleDetailScreen(article: article);
      },
    ),
  ],
);
