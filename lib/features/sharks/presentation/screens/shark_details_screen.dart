import 'package:flutter/material.dart';
import '../../../../core/theme/app_theme.dart';
import '../../domain/models/shark.dart';
import '../widgets/shark_trajectory_map.dart';
import '../widgets/shark_event_item.dart';

class SharkDetailsScreen extends StatelessWidget {
  final Shark shark;

  const SharkDetailsScreen({super.key, required this.shark});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: DefaultTabController(
        length: 3,
        child: NestedScrollView(
          headerSliverBuilder: (context, innerBoxIsScrolled) {
            return [
              SliverAppBar(
                expandedHeight: 300.0,
                floating: false,
                pinned: true,
                backgroundColor: AppTheme.deepOceanBlue,
                leading: IconButton(
                  icon: const Icon(
                    Icons.arrow_back_ios_new,
                    color: Colors.white,
                  ),
                  onPressed: () => Navigator.of(context).pop(),
                ),
                flexibleSpace: FlexibleSpaceBar(
                  centerTitle: true,
                  title: Text(
                    shark.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      shadows: [Shadow(color: Colors.black45, blurRadius: 10)],
                    ),
                  ),
                  background: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.network(shark.imageUrl, fit: BoxFit.cover),
                      const DecoratedBox(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [Colors.transparent, Colors.black54],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SliverPersistentHeader(
                delegate: _SliverAppBarDelegate(
                  const TabBar(
                    labelColor: AppTheme.deepOceanBlue,
                    unselectedLabelColor: Colors.grey,
                    indicatorColor: AppTheme.oceanBlue,
                    indicatorWeight: 3,
                    tabs: [
                      Tab(text: "Trayectoria"),
                      Tab(text: "Eventos"),
                      Tab(text: "Bio-Datos"),
                    ],
                  ),
                ),
                pinned: true,
              ),
            ];
          },
          body: TabBarView(
            children: [
              SharkTrajectoryMap(shark: shark),
              ListView(
                padding: const EdgeInsets.all(24),
                children: const [
                  SharkEventItem(
                    title: "Avistamiento en Superficie",
                    date: "Hace 2 horas",
                    description:
                        "El tiburón emergió brevemente cerca de la costa, probablemente buscando aguas más cálidas.",
                    imageUrl:
                        "https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&q=80",
                  ),
                  SharkEventItem(
                    title: "Evento de Caza Exitosa",
                    date: "Ayer, 18:30",
                    description:
                        "Se detectó un patrón de aceleración rápido indicando una persecución exitosa de cardumen.",
                    imageUrl:
                        "https://images.unsplash.com/photo-1582967788606-a171f1080ca8?auto=format&fit=crop&q=80", // Reliable ocean/predator image
                  ),
                  SharkEventItem(
                    title: "Marcaje Inicial",
                    date: "Hace 15 días",
                    description:
                        "El equipo de BlueSentinel instaló el rastreador satelital sin complicaciones.",
                    imageUrl:
                        "https://images.unsplash.com/photo-1599157262102-5ee3a7cf5273?auto=format&fit=crop&q=80",
                    isLast: true,
                  ),
                ],
              ),
              _buildBioDataTab(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBioDataTab(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        _buildDetailItem(context, "Especie", shark.species),
        _buildDetailItem(context, "Nombre Científico", shark.scientificName),
        _buildDetailItem(context, "Estado", shark.status),
        _buildDetailItem(context, "Edad", "${shark.age} años"),
        _buildDetailItem(context, "Longitud", "${shark.length} m"),
        _buildDetailItem(context, "Peso", shark.weight),
        _buildDetailItem(
          context,
          "Última Actualización",
          shark.lastUpdate.toString().split(' ')[0],
        ),
      ],
    );
  }

  Widget _buildDetailItem(BuildContext context, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[600], fontSize: 16)),
          Text(
            value,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: AppTheme.deepOceanBlue,
            ),
          ),
        ],
      ),
    );
  }
}

class _SliverAppBarDelegate extends SliverPersistentHeaderDelegate {
  final TabBar _tabBar;

  _SliverAppBarDelegate(this._tabBar);

  @override
  double get minExtent => _tabBar.preferredSize.height;
  @override
  double get maxExtent => _tabBar.preferredSize.height;

  @override
  Widget build(
    BuildContext context,
    double shrinkOffset,
    bool overlapsContent,
  ) {
    return Container(color: Colors.white, child: _tabBar);
  }

  @override
  bool shouldRebuild(_SliverAppBarDelegate oldDelegate) {
    return false;
  }
}
