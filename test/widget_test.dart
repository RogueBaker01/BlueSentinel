import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:bluesentinel/main.dart';

void main() {
  testWidgets('App initializes smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ProviderScope(child: BlueSentinelApp()));

    // Verify that Splash screen appears initially (BlueSentinel title)
    expect(find.text('BlueSentinel'), findsOneWidget);
  });
}
