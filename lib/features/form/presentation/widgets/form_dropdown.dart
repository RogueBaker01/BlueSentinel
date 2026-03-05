import 'package:bluesentinel/core/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

//  FORM DROPDOWN  
class FormDropdown extends StatelessWidget {
  final String? value;
  final List<String> items;
  final String hint;
  final ValueChanged<String?> onChanged;
  final String? Function(String?)? validator;

  const FormDropdown({super.key, this.value, required this.items, required this.hint, required this.onChanged, this.validator});

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField<String>(
      initialValue: value,
      hint: Text(hint,
          style: GoogleFonts.inter(color: AppTheme.textMuted, fontSize: 14)),
      icon: const Icon(Icons.keyboard_arrow_down_rounded, color: AppTheme.accentTurquoise),
      style: GoogleFonts.inter(
          color: Colors.white, fontSize: 15, fontWeight: FontWeight.w500),
      dropdownColor: AppTheme.darkBlue,
      borderRadius: BorderRadius.circular(14),
      decoration: InputDecoration(
        prefixIcon:
            const Icon(Icons.badge_outlined, color: AppTheme.accentTurquoise, size: 20),
        filled: true,
        fillColor: AppTheme.cardBg,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppTheme.borderColor),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppTheme.borderColor),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppTheme.accentTurquoise, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppTheme.alertRed, width: 1.5),
        ),
        errorStyle: const TextStyle(color: AppTheme.alertRed, fontSize: 12),
      ),
      items: items
          .map((r) => DropdownMenuItem(value: r, child: Text(r)))
          .toList(),
      onChanged: onChanged,
      validator: validator,
    );
  }
}

