import pymeshlab

ms = pymeshlab.MeshSet()

try:
    # Load file gốc
    print("Loading original mesh...")
    ms.load_new_mesh('rp_dennis_posed_004_30k.obj')

    # Chuyển tất cả mặt về triangle nếu chưa phải
    print("Converting quads to triangles...")
    ms.apply_filter('quads_to_triangles')

    # Loop subdivision 1-2 lần
    print("Applying loop subdivision...")
    ms.apply_filter('loop_subdivision', iterations=2)

    # Xuất file mới
    print("Saving the new mesh...")
    ms.save_current_mesh('rp_dennis_posed_004_100k.obj')

    print("✅ Xong! File mới đã tăng mặt từ 30k lên ~100k.")

except pymeshlab.pmeshlab.PyMeshLabException as e:
    print(f"Lỗi: {str(e)}")
    # In ra danh sách các filter có sẵn
    print("\nDanh sách các filter có sẵn:")
    filters = [f for f in dir(ms) if not f.startswith('_')]
    print('\n'.join(filters))

print([f for f in dir(ms) if 'apply_filter' in f])
