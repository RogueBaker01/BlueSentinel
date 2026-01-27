import '../../domain/models/shark.dart';

class SharkRepository {
  Future<List<Shark>> getSharks() async {
    // Simular delay de red
    await Future.delayed(const Duration(milliseconds: 800));

    return [
      Shark(
        id: '1',
        name: 'Titan',
        species: 'Great White Shark',
        scientificName: 'Carcharodon carcharias',
        imageUrl:
            'https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80',
        status: 'Active',
        lastLatitude: 28.9,
        lastLongitude: -118.2,
        lastUpdate: DateTime.now().subtract(const Duration(hours: 5)),
        age: 15,
        length: 4.5,
        weight: '900 kg',
      ),
      Shark(
        id: '2',
        name: 'Luna',
        species: 'Whale Shark',
        scientificName: 'Rhincodon typus',
        imageUrl:
            'https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&q=80', // Fallback to reliable whale shark image
        status: 'Active',
        lastLatitude: 21.5,
        lastLongitude: -87.1,
        lastUpdate: DateTime.now().subtract(const Duration(hours: 2)),
        age: 8,
        length: 8.2,
        weight: '12,000 kg',
      ),
      Shark(
        id: '3',
        name: 'Shadow',
        species: 'Hammerhead Shark',
        scientificName: 'Sphyrna mokarran',
        imageUrl:
            'https://images.unsplash.com/photo-1563811776595-32e56230f287?auto=format&fit=crop&q=80',
        status: 'Active',
        lastLatitude: 5.5,
        lastLongitude: -87.0,
        lastUpdate: DateTime.now().subtract(const Duration(days: 1)),
        age: 10,
        length: 3.8,
        weight: '350 kg',
      ),
      Shark(
        id: '4',
        name: 'Blue',
        species: 'Blue Shark',
        scientificName: 'Prionace glauca',
        imageUrl:
            'https://images.unsplash.com/photo-1564619448883-7c0936e7884f?auto=format&fit=crop&q=80',
        status: 'Lost',
        lastLatitude: 40.2,
        lastLongitude: -30.5,
        lastUpdate: DateTime.now().subtract(const Duration(days: 45)),
        age: 6,
        length: 2.5,
        weight: '110 kg',
      ),
    ];
  }
}
