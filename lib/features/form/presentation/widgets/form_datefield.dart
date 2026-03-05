import 'package:bluesentinel/core/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

//  SELECTOR DE FECHA 
class FormDatefield extends StatelessWidget {
  final DateTime? fecha;
  final VoidCallback onTap;

  const FormDatefield({super.key, this.fecha, required this.onTap});

  String get _label {
    if (fecha == null) return 'Selecciona una fecha';
    return '${fecha!.day.toString().padLeft(2, '0')}/'
        '${fecha!.month.toString().padLeft(2, '0')}/'
        '${fecha!.year}';
  }

  @override
  Widget build(BuildContext context) {
    final hasValue = fecha != null;
    return FormField<DateTime>(
      initialValue: fecha,
      validator: (_) =>
          fecha == null ? 'Selecciona tu fecha de nacimiento' : null,
      builder: (state) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: onTap,
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 16),
              decoration: BoxDecoration(
                color: AppTheme.cardBg,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: state.hasError ? AppTheme.alertRed : AppTheme.borderColor,
                  width: state.hasError ? 1.5 : 1,
                ),
              ),
              child: Row(
                children: [
                  const Icon(Icons.calendar_today_outlined,
                      color: AppTheme.accentTurquoise, size: 20),
                  const SizedBox(width: 12),
                  Text(
                    _label,
                    style: GoogleFonts.inter(
                      color: hasValue ? Colors.white : AppTheme.textMuted,
                      fontSize: hasValue ? 15 : 14,
                      fontWeight: hasValue
                          ? FontWeight.w500
                          : FontWeight.normal,
                    ),
                  ),
                  const Spacer(),
                  const Icon(Icons.arrow_forward_ios_rounded,
                      size: 13, color: AppTheme.textMuted),
                ],
              ),
            ),
          ),
          if (state.hasError)
            Padding(
              padding: const EdgeInsets.only(top: 6, left: 12),
              child: Text(
                state.errorText!,
                style: const TextStyle(color: AppTheme.alertRed, fontSize: 12),
              ),
            ),
        ],
      ),
    );
  }
}

