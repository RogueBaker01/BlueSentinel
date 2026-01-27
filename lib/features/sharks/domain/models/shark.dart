class Shark {
  final String id;
  final String name;
  final String species;
  final String scientificName;
  final String imageUrl;
  final String status; // 'Active', 'Lost', 'Inactive'
  final double lastLatitude;
  final double lastLongitude;
  final DateTime lastUpdate;
  final int age;
  final double length; // Meters
  final String weight; // kg

  const Shark({
    required this.id,
    required this.name,
    required this.species,
    required this.scientificName,
    required this.imageUrl,
    required this.status,
    required this.lastLatitude,
    required this.lastLongitude,
    required this.lastUpdate,
    required this.age,
    required this.length,
    required this.weight,
  });
}
