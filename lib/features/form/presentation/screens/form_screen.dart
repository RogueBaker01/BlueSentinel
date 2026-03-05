// El form deberá de cumplir con los siguientes requisitos:
// -- Debe implementar un Form con una GlobalKey<FormState>.
// -- Campos obligatorios con validación:
// -- Al menos un campo de texto (ej. Nombre, Título).
// -- Al menos un campo numérico con teclado específico (ej. Precio, Edad, Cantidad).
// -- Un selector de opciones (DropdownButtonFormField).
// -- Un selector de fecha nativo (showDatePicker).
// -- Feedback: Al presionar el botón de guardar, la app debe validar los datos y mostrar un SnackBar confirmando la acción.

import 'package:bluesentinel/core/router/app_router.dart';
import 'package:bluesentinel/core/theme/app_theme.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_datefield.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_dropdown.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_header.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_loginbtn.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_textfield.dart';
import 'package:bluesentinel/features/form/presentation/widgets/ocean_background.dart';
import 'package:bluesentinel/features/form/presentation/widgets/form_socialsection.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';


class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen>
    with SingleTickerProviderStateMixin {
  // ── Form key ──────────────────────────────
  final _formKey = GlobalKey<FormState>();

  // ── Controllers ───────────────────────────
  final _nombreCtrl = TextEditingController();
  final _edadCtrl   = TextEditingController();

  // ── State ─────────────────────────────────
  String?   _rolSeleccionado;
  DateTime? _fechaNacimiento;
  bool      _obscurePass = true;

  // ── Animación ─────────────────────────────
  late AnimationController _animCtrl;
  late Animation<double>   _fadeAnim;
  late Animation<Offset>   _slideAnim;

  static const _roles = [
    'Administrador',
    'Investigador',
    'Observador',
    'Invitado',
  ];

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.10),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _animCtrl, curve: Curves.easeOutCubic));
    _animCtrl.forward();
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _nombreCtrl.dispose();
    _edadCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final hoy = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _fechaNacimiento ?? DateTime(hoy.year - 18),
      firstDate: DateTime(1920),
      lastDate: hoy,
      builder: (context, child) => Theme(
        data: Theme.of(context).copyWith(
          colorScheme: const ColorScheme.dark(
            primary: AppTheme.accentTurquoise,
            onPrimary: AppTheme.deepOceanBlue,
            surface: AppTheme.darkBlue,
            onSurface: Colors.white,
          ), dialogTheme: DialogThemeData(backgroundColor: AppTheme.deepOceanBlue),
        ),
        child: child!,
      ),
    );
    if (picked != null) setState(() => _fechaNacimiento = picked);
  }

  void _onGuardar() {
    if (!_formKey.currentState!.validate()) return;
    _formKey.currentState!.save();

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.all(16),
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        backgroundColor: AppTheme.darkBlue,
        content: Row(
          children: [
            const Icon(Icons.check_circle_rounded, color: AppTheme.green, size: 20),
            const SizedBox(width: 10),
            Text(
              '¡Bienvenido, ${_nombreCtrl.text}!',
              style: GoogleFonts.inter(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        duration: const Duration(seconds: 2),
      ),
    );

    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      appRouter.go('/home');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.deepOceanBlue,
      body: Stack(
        children: [
          OceanBackground(),
          SafeArea(
            child: FadeTransition(
              opacity: _fadeAnim,
              child: SlideTransition(
                position: _slideAnim,
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 24, vertical: 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 8),
                      FormHeader(),
                      const SizedBox(height: 32),
                      _buildFormCard(),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFormCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.darkBlue,
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: AppTheme.borderColor),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.35),
            blurRadius: 40,
            offset: const Offset(0, 16),
          ),
          BoxShadow(
            color: AppTheme.accentTurquoise.withValues(alpha:0.05),
            blurRadius: 60,
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ① Nombre
              _fieldLabel('Nombre completo'),
              const SizedBox(height: 8),
              FormTextfield(
                controller: _nombreCtrl,
                hint: 'Ej. Ana García',
                icon: Icons.person_outline_rounded,
                keyboardType: TextInputType.number,
                obscureText: false,
                validator: (v) {
                  if (v == null || v.trim().isEmpty) {
                    return 'El nombre es obligatorio';
                  }
                  if (v.trim().length < 3) return 'Mínimo 3 caracteres';
                  return null;
                },
              ),

              const SizedBox(height: 20),

              // ② Edad (numérico)
              _fieldLabel('Edad'),
              const SizedBox(height: 8),
              FormTextfield(
                controller: _edadCtrl,
                hint: 'Ej. 28',
                icon: Icons.tag_rounded,
                keyboardType: TextInputType.number,
                obscureText: false,
                validator: (v) {
                  if (v == null || v.trim().isEmpty) {
                    return 'La edad es obligatoria';
                  }
                  final n = int.tryParse(v.trim());
                  if (n == null) return 'Ingresa un número válido';
                  if (n < 1 || n > 120) return 'Edad entre 1 y 120';
                  return null;
                },
              ),
              const SizedBox(height: 20),

              // ③ Rol (Dropdown)
              _fieldLabel('Rol en la plataforma'),
              const SizedBox(height: 8),
              FormDropdown(
                value: _rolSeleccionado,
                items: _roles,
                hint: 'Selecciona un rol',
                onChanged: (v) => setState(() => _rolSeleccionado = v),
                validator: (v) => v == null ? 'Selecciona un rol' : null,
              ),

              const SizedBox(height: 20),

              // ④ Fecha de nacimiento
              _fieldLabel('Fecha de nacimiento'),
              const SizedBox(height: 8),
              FormDatefield(
                fecha: _fechaNacimiento,
                onTap: _pickDate,
              ),

              const SizedBox(height: 20),

              // ⑤ Contraseña
              _fieldLabel('Contraseña'),
              const SizedBox(height: 8),
              FormTextfield(
                hint: '••••••••',
                icon: Icons.lock_outline_rounded,
                keyboardType: TextInputType.number,
                obscureText: _obscurePass,
                suffixIcon: IconButton(
                  icon: Icon(
                    _obscurePass
                        ? Icons.visibility_off_outlined
                        : Icons.visibility_outlined,
                    color: AppTheme.textMuted,
                    size: 20,
                  ),
                  onPressed: () =>
                      setState(() => _obscurePass = !_obscurePass),
                ),
                validator: (v) {
                  if (v == null || v.isEmpty) {
                    return 'La contraseña es obligatoria';
                  }
                  if (v.length < 6) return 'Mínimo 6 caracteres';
                  return null;
                },
              ),

              const SizedBox(height: 32),

              FormLoginbtn(onPressed: _onGuardar),

              const SizedBox(height: 20),

              FormSocialsection(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _fieldLabel(String text) => Text(
        text,
        style: GoogleFonts.inter(
          color: AppTheme.textMuted,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ),
      );
}

