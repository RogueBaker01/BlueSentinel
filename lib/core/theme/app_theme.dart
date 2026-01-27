import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Colores Oceánicos
  static const Color deepOceanBlue = Color(0xFF001E3C);
  static const Color oceanBlue = Color(0xFF023E8A);
  static const Color surfaceBlue = Color(0xFFE0F7FA);
  static const Color accentTurquoise = Color(0xFF00B4D8);
  static const Color alertRed = Color(0xFFFF5252);
  static const Color conservationGreen = Color(0xFF69F0AE);

  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      colorScheme: ColorScheme.fromSeed(
        seedColor: oceanBlue,
        brightness: Brightness.light,
        primary: deepOceanBlue,
        secondary: accentTurquoise,
        surface: Colors.white,
        error: alertRed,
      ),
      scaffoldBackgroundColor: Colors.white,

      // Tipografía
      textTheme: GoogleFonts.interTextTheme().copyWith(
        displayLarge: GoogleFonts.inter(
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: deepOceanBlue,
        ),
        displayMedium: GoogleFonts.inter(
          fontSize: 24,
          fontWeight: FontWeight.bold,
          color: deepOceanBlue,
        ),
        bodyLarge: GoogleFonts.lato(fontSize: 16, color: Colors.black87),
        bodyMedium: GoogleFonts.lato(fontSize: 14, color: Colors.black87),
      ),

      // AppBar Theme
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          color: deepOceanBlue,
          fontSize: 18,
          fontWeight: FontWeight.w600,
        ),
        iconTheme: const IconThemeData(color: deepOceanBlue),
      ),

      // Card Theme
      cardTheme: CardThemeData(
        elevation: 2,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        color: surfaceBlue.withValues(alpha: 0.3),
      ),
    );
  }

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: ColorScheme.fromSeed(
        seedColor: oceanBlue,
        brightness: Brightness.dark,
        primary: accentTurquoise,
        secondary: conservationGreen,
        surface: const Color(0xFF0A1929),
        error: alertRed,
      ),
      scaffoldBackgroundColor: const Color(0xFF000E21),

      // Tipografía Dark
      textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme)
          .copyWith(
            displayLarge: GoogleFonts.inter(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),

      appBarTheme: AppBarTheme(
        backgroundColor: const Color(0xFF000E21),
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.w600,
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
    );
  }
}
