import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';
import '../../data/repositories/shark_repository.dart';
import '../../domain/models/shark.dart';
import '../widgets/shark_card.dart';

class SharksScreen extends StatefulWidget {
  const SharksScreen({super.key});

  @override
  State<SharksScreen> createState() => _SharksScreenState();
}

class _SharksScreenState extends State<SharksScreen> {
  final SharkRepository _repository = SharkRepository();
  List<Shark> _allSharks = [];
  List<Shark> _filteredSharks = [];
  bool _isLoading = true;
  bool _isSearching = false;
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadSharks();
  }

  Future<void> _loadSharks() async {
    final sharks = await _repository.getSharks();
    if (mounted) {
      setState(() {
        _allSharks = sharks;
        _filteredSharks = sharks;
        _isLoading = false;
      });
    }
  }

  void _runFilter(String query) {
    setState(() {
      if (query.isEmpty) {
        _filteredSharks = _allSharks;
      } else {
        _filteredSharks = _allSharks
            .where(
              (shark) =>
                  shark.name.toLowerCase().contains(query.toLowerCase()) ||
                  shark.species.toLowerCase().contains(query.toLowerCase()),
            )
            .toList();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: _isSearching
            ? TextField(
                controller: _searchController,
                autofocus: true,
                decoration: const InputDecoration(
                  hintText: 'Buscar tiburón...',
                  border: InputBorder.none,
                  hintStyle: TextStyle(color: Colors.grey),
                ),
                style: const TextStyle(color: AppTheme.deepOceanBlue),
                onChanged: _runFilter,
              )
            : const Text('Tracked Sharks'),
        backgroundColor: Colors.white,
        centerTitle: false,
        elevation: 0,
        titleTextStyle: const TextStyle(
          color: AppTheme.deepOceanBlue,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        actions: [
          IconButton(
            icon: Icon(
              _isSearching ? Icons.close : Icons.search,
              color: AppTheme.deepOceanBlue,
            ),
            onPressed: () {
              setState(() {
                if (_isSearching) {
                  _isSearching = false;
                  _searchController.clear();
                  _runFilter('');
                } else {
                  _isSearching = true;
                }
              });
            },
          ),
          if (!_isSearching)
            IconButton(
              icon: const Icon(
                Icons.filter_list,
                color: AppTheme.deepOceanBlue,
              ),
              onPressed: () {
                _showFilterModal();
              },
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _filteredSharks.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.search_off, size: 64, color: Colors.grey[400]),
                  const SizedBox(height: 16),
                  Text(
                    'No se encontraron tiburones',
                    style: TextStyle(color: Colors.grey[600]),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _filteredSharks.length,
              itemBuilder: (context, index) {
                final shark = _filteredSharks[index];
                return SharkCard(
                  shark: shark,
                  onTap: () {
                    context.push('/shark-details', extra: shark);
                  },
                );
              },
            ),
    );
  }

  void _showFilterModal() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Filtrar por Especies',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppTheme.deepOceanBlue,
                ),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                children:
                    [
                      'Great White Shark',
                      'Whale Shark',
                      'Hammerhead Shark',
                      'Blue Shark',
                    ].map((species) {
                      return FilterChip(
                        label: Text(species),
                        onSelected: (selected) {
                          Navigator.pop(context);
                          _runFilter(species);
                          _searchController.text = species;
                          setState(() => _isSearching = true);
                        },
                      );
                    }).toList(),
              ),
            ],
          ),
        );
      },
    );
  }
}
