import 'package:bluesentinel/core/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

//  HEADER
class FormHeader extends StatelessWidget {
  const FormHeader({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppTheme.accentTurquoise.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.accentTurquoise.withValues(alpha: 0.3)),
              ),
              child: Image.asset( 
                'assets/images/logo.png',
                width: 24,
                height: 24,
                fit: BoxFit.contain,),
            ),
            const SizedBox(width: 12),
            Text(
              'BlueSentinel',
              style: GoogleFonts.inter(
                color: AppTheme.accentTurquoise,
                fontSize: 13,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),
        Text(
          'Crea tu\nperfil',
          style: GoogleFonts.inter(
            color: Colors.white,
            fontSize: 40,
            fontWeight: FontWeight.w800,
            height: 1.1,
            letterSpacing: -1.2,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Completa los datos para acceder\na la plataforma',
          style: GoogleFonts.lato(
            color: AppTheme.textMuted,
            fontSize: 15,
            height: 1.5,
          ),
        ),
      ],
    );
  }
}

