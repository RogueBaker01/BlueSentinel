import 'package:bluesentinel/core/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

//  SECCIÓN SOCIAL
class FormSocialsection extends StatelessWidget {
  const FormSocialsection({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            const Expanded(
                child: Divider(color: AppTheme.borderColor, thickness: 1)),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Text(
                'o inicia con',
                style: GoogleFonts.inter(color: AppTheme.textMuted, fontSize: 13),
              ),
            ),
            const Expanded(
                child: Divider(color: AppTheme.borderColor, thickness: 1)),
          ],
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: SocialBtn(
                  label: 'Google', textColor: const Color(0xFFEA4335)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: SocialBtn(
                  label: 'GitHub', textColor: Colors.white),
            ),
          ],
        ),
      ],
    );
  }
}

class SocialBtn extends StatelessWidget {
  final String label;
  final Color textColor;

  const SocialBtn({super.key, required this.label, required this.textColor});



  @override
  Widget build(BuildContext context) {
    return Container(
      height: 48,
      decoration: BoxDecoration(
        color: AppTheme.cardBg,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.borderColor),
      ),
      child: Center(
        child: Text(
          label,
          style: GoogleFonts.inter(
            color: textColor,
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
      ),
    );
  }
}