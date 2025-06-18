import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup


# Configuration for a GPU resource
gpu_config = LiveServerless(
    gpus=[
        GpuGroup.ADA_24,
        GpuGroup.AMPERE_48,
    ],
    name="example_protein_folding",
)


@remote(
    resource_config=gpu_config,
    dependencies=["torch"],
)
def simulate_protein_folding():
    import torch

    # Define a simplified energy function for protein folding
    def energy_function(positions):
        # Harmonic potential to keep atoms close to their ideal distances
        bond_energy = (
            torch.sum((positions[1:] - positions[:-1]).pow(2).sum(dim=1) - 1.0) ** 2
        )
        # Repulsive potential to prevent atoms from overlapping
        repulsion_energy = torch.sum(
            1.0 / (torch.norm(positions[:, None] - positions, dim=2).pow(12) + 1e-6)
        )
        return bond_energy + repulsion_energy

    # Initialize random 3D positions for a chain of 10 "atoms"
    positions = torch.randn(10, 3, device="cuda", requires_grad=True)

    # Optimizer for minimizing the energy
    optimizer = torch.optim.Adam([positions], lr=0.01)

    # Perform gradient descent to minimize the energy
    for step in range(1000):
        optimizer.zero_grad()
        energy = energy_function(positions)
        energy.backward()
        optimizer.step()

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Energy: {energy.item()}")

    return {
        "final_positions": positions.tolist(),
        "final_energy": energy.item(),
    }


async def main():
    print("\nSimulating protein folding...")
    folding_result = await simulate_protein_folding()
    print(f"\nProtein Folding Result: {folding_result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
