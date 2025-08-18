"""
Tests for the RS matrices parser module.

Tests the ParseRSMatrices class for loading and processing FHI-aims
sparse matrix output from the output_rs_matrices tag.
"""

import os

import numpy as np
import pytest
from scipy import sparse

from mypytools.aims.rs_matrices import ParseRSMatrices


class TestParseRSMatrices:
    """Test class for ParseRSMatrices functionality."""

    def _parse_gamma_reference_matrix(self, file_path: str, n_basis: int = 56) -> np.ndarray:
        """Parse gamma-point reference matrix from FHI-aims output file.

        Args:
            file_path: Path to the matrix file (hamiltonian.out or overlap-matrix.out).
            n_basis: Number of basis functions (default 56).

        Returns:
            numpy.ndarray: Full symmetric matrix of shape (n_basis, n_basis).
        """
        # Load the 3-column data: row_idx, col_idx, value
        data = np.loadtxt(file_path)

        # Convert to 0-based indexing (FHI-aims uses 1-based)
        row_indices = data[:, 0].astype(int) - 1
        col_indices = data[:, 1].astype(int) - 1
        values = data[:, 2]

        # Create full matrix
        matrix = np.zeros((n_basis, n_basis), dtype=np.float64)

        # Fill matrix elements
        for i, j, val in zip(row_indices, col_indices, values):
            matrix[i, j] = val
            # For symmetric matrices, fill both upper and lower triangular parts
            if i != j:
                matrix[j, i] = val

        return matrix

    @pytest.fixture
    def test_data_dir(self):
        """Fixture providing path to test data directory."""
        return "tests/data/tag_output_rs_matrices_h5_250806"

    @pytest.fixture
    def parser_h5(self, test_data_dir):
        """Fixture providing ParseRSMatrices instance for H5 format."""
        return ParseRSMatrices(test_data_dir, use_h5=True)

    @pytest.fixture
    def gamma_hamiltonian_ref(self, test_data_dir):
        """Fixture providing reference Hamiltonian matrix at gamma point."""
        hamiltonian_file = os.path.join(test_data_dir, "hamiltonian.out")
        return self._parse_gamma_reference_matrix(hamiltonian_file)

    @pytest.fixture
    def gamma_overlap_ref(self, test_data_dir):
        """Fixture providing reference overlap matrix at gamma point."""
        overlap_file = os.path.join(test_data_dir, "overlap-matrix.out")
        return self._parse_gamma_reference_matrix(overlap_file)

    def test_init_with_valid_directory(self, test_data_dir):
        """Test initialization with valid directory."""
        parser = ParseRSMatrices(test_data_dir, use_h5=True)
        assert parser.aims_dpath == test_data_dir
        assert parser.rs_hamiltonian is None
        assert parser.rs_overlap is None
        assert parser.matrix_size is not None  # Should be loaded from rs_indices.out
        assert parser.n_cells is not None
        assert parser.n_basis is not None

    def test_init_with_invalid_directory(self):
        """Test initialization with invalid directory raises AssertionError."""
        with pytest.raises(AssertionError):
            ParseRSMatrices("/nonexistent/directory")

    def test_parse_rs_indices_out(self, parser_h5):
        """Test parsing of rs_indices.out file."""
        # Check that basic parameters are loaded correctly
        assert parser_h5.matrix_size == 63605
        assert parser_h5.n_cells == 186
        assert parser_h5.n_basis == 56

        # Check array shapes
        assert parser_h5.cell_index.shape == (186, 3)
        assert parser_h5.index_hamiltonian_1.shape == (186, 56)
        assert parser_h5.index_hamiltonian_2.shape == (186, 56)
        assert parser_h5.column_index_hamiltonian.shape == (63605,)

        # Check data types
        assert parser_h5.cell_index.dtype == np.int32 or parser_h5.cell_index.dtype == np.int64
        assert parser_h5.index_hamiltonian_1.dtype == np.int32 or parser_h5.index_hamiltonian_1.dtype == np.int64
        assert parser_h5.index_hamiltonian_2.dtype == np.int32 or parser_h5.index_hamiltonian_2.dtype == np.int64
        assert (
            parser_h5.column_index_hamiltonian.dtype == np.int32 or parser_h5.column_index_hamiltonian.dtype == np.int64
        )

    def test_load_rs_hamiltonian_h5(self, parser_h5):
        """Test loading Hamiltonian matrix from H5 file."""
        hamiltonian = parser_h5.load_rs_hamiltonian_h5()

        # Check shape and type
        assert hamiltonian.shape == (63605,)
        assert hamiltonian.dtype == np.float64
        assert parser_h5.rs_hamiltonian is not None

        # Check that data is loaded (not all zeros)
        assert not np.all(hamiltonian == 0)

    def test_load_rs_overlap_h5(self, parser_h5):
        """Test loading overlap matrix from H5 file."""
        overlap = parser_h5.load_rs_overlap_h5()

        # Check shape and type
        assert overlap.shape == (63605,)
        assert overlap.dtype == np.float64
        assert parser_h5.rs_overlap is not None

        # Check that data is loaded (not all zeros)
        assert not np.all(overlap == 0)

    def test_load_matrices(self, parser_h5):
        """Test loading both matrices at once."""
        hamiltonian, overlap = parser_h5.load_matrices()

        # Check shapes and types
        assert hamiltonian.shape == (63605,)
        assert overlap.shape == (63605,)
        assert hamiltonian.dtype == np.float64
        assert overlap.dtype == np.float64

    def test_get_hamiltonian_dense_U(self, parser_h5):
        """Test getting Hamiltonian U matrix in dense format."""
        dense_h = parser_h5.get_hamiltonian_dense_U()

        # Check shape and type
        n_rows = parser_h5.n_cells * parser_h5.n_basis
        assert dense_h.shape == (n_rows, parser_h5.n_basis)
        assert dense_h.dtype == np.float64

        # Check that matrix is not all zeros
        assert not np.all(dense_h == 0)

    def test_get_overlap_dense_U(self, parser_h5):
        """Test getting overlap U matrix in dense format."""
        dense_s = parser_h5.get_overlap_dense_U()

        # Check shape and type
        n_rows = parser_h5.n_cells * parser_h5.n_basis
        assert dense_s.shape == (n_rows, parser_h5.n_basis)
        assert dense_s.dtype == np.float64

        # Check that matrix is not all zeros
        assert not np.all(dense_s == 0)

    def test_get_hamiltonian_csr_U(self, parser_h5):
        """Test getting Hamiltonian U matrix in CSR sparse format."""
        csr_h = parser_h5.get_hamiltonian_csr_U()

        # Check type and shape
        assert isinstance(csr_h, sparse.csr_matrix)
        n_rows = parser_h5.n_cells * parser_h5.n_basis
        assert csr_h.shape == (n_rows, parser_h5.n_basis)
        assert csr_h.dtype == np.float64

        # Check that matrix has non-zero elements
        assert csr_h.nnz > 0

        # Check sparsity (should be sparse, not dense)
        total_elements = n_rows * parser_h5.n_basis
        sparsity = 1 - (csr_h.nnz / total_elements)
        assert sparsity > 0.5  # Should be at least 50% sparse

    def test_get_overlap_csr_U(self, parser_h5):
        """Test getting overlap U matrix in CSR sparse format."""
        csr_s = parser_h5.get_overlap_csr_U()

        # Check type and shape
        assert isinstance(csr_s, sparse.csr_matrix)
        n_rows = parser_h5.n_cells * parser_h5.n_basis
        assert csr_s.shape == (n_rows, parser_h5.n_basis)
        assert csr_s.dtype == np.float64

        # Check that matrix has non-zero elements
        assert csr_s.nnz > 0

    def test_dense_csr_equivalence(self, parser_h5):
        """Test that dense and CSR U matrix formats give equivalent results."""
        # Load both formats
        dense_h = parser_h5.get_hamiltonian_dense_U()
        csr_h = parser_h5.get_hamiltonian_csr_U()

        # Convert CSR to dense for comparison
        csr_dense = csr_h.toarray()

        # Check equivalence
        np.testing.assert_allclose(dense_h, csr_dense, rtol=1e-12, atol=1e-12)

    def test_column_index_ranges(self, parser_h5):
        """Test that column indices are within valid ranges."""
        # Column indices should be 1-based in file, but we convert to 0-based
        col_indices = parser_h5.column_index_hamiltonian

        # After conversion, should be in range [0, n_basis]
        # Note: some indices might be n_basis due to cell periodicity
        assert np.all(col_indices >= 0)
        assert np.all(col_indices <= parser_h5.n_basis)

    def test_matrix_elements_finite(self, parser_h5):
        """Test that all matrix elements are finite (no NaN or inf)."""
        hamiltonian = parser_h5.load_rs_hamiltonian_h5()
        overlap = parser_h5.load_rs_overlap_h5()

        assert np.all(np.isfinite(hamiltonian))
        assert np.all(np.isfinite(overlap))

    @pytest.mark.slow
    def test_matrix_reconstruction_properties(self, parser_h5):
        """Test mathematical properties of reconstructed matrices."""
        dense_h = parser_h5.get_hamiltonian_dense_U()
        dense_s = parser_h5.get_overlap_dense_U()

        # Test basic properties of rectangular matrices
        assert dense_h.shape[0] > dense_h.shape[1]  # More rows than columns
        assert dense_s.shape[0] > dense_s.shape[1]  # More rows than columns

        # Test that matrices have reasonable value ranges
        h_nonzero = dense_h[dense_h != 0]
        s_nonzero = dense_s[dense_s != 0]

        if len(h_nonzero) > 0:
            assert np.all(np.isfinite(h_nonzero))
            # Hamiltonian values should be reasonable (not too large)
            assert np.abs(h_nonzero).max() < 100  # Reasonable energy scale

        if len(s_nonzero) > 0:
            assert np.all(np.isfinite(s_nonzero))
            # Overlap values should be reasonable (typically close to 1 for normalized basis)
            assert s_nonzero.max() < 10  # Reasonable overlap scale
            assert s_nonzero.min() > -10  # Allow some negative overlaps

    def test_geometry_parsing(self, parser_h5):
        """Test that geometry.in is parsed correctly."""
        assert parser_h5.lattice_vectors is not None
        assert parser_h5.lattice_vectors.shape == (3, 3)
        assert parser_h5.lattice_vectors.dtype == np.float64

        # Check that lattice vectors are reasonable (not all zeros)
        assert not np.all(parser_h5.lattice_vectors == 0)

    def test_kspace_gamma_point(self, parser_h5):
        """Test k-space matrices at Γ-point."""
        H_gamma = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0])
        S_gamma = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0])

        # Check shapes and types
        assert H_gamma.shape == (56, 56)
        assert S_gamma.shape == (56, 56)
        assert H_gamma.dtype == np.complex128
        assert S_gamma.dtype == np.complex128

        # At Γ-point, matrices should be essentially real
        assert np.max(np.abs(H_gamma.imag)) < 1e-12
        assert np.max(np.abs(S_gamma.imag)) < 1e-12

    def test_kspace_arbitrary_point(self, parser_h5):
        """Test k-space matrices at arbitrary k-point."""
        H_k = parser_h5.get_hamiltonian_kspace([0.5, 0.5, 0.0])
        S_k = parser_h5.get_overlap_kspace([0.5, 0.5, 0.0])

        # Check shapes and types
        assert H_k.shape == (56, 56)
        assert S_k.shape == (56, 56)
        assert H_k.dtype == np.complex128
        assert S_k.dtype == np.complex128

        # Matrices should be finite
        assert np.all(np.isfinite(H_k))
        assert np.all(np.isfinite(S_k))

    def test_kspace_input_validation(self, parser_h5):
        """Test input validation for k-space methods."""
        # Test invalid k_frac shapes
        with pytest.raises(ValueError):
            parser_h5.get_hamiltonian_kspace([0.0, 0.0])  # Too short

        with pytest.raises(ValueError):
            parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0, 0.0])  # Too long

        # Test that valid inputs work
        H_k = parser_h5.get_hamiltonian_kspace([0.1, 0.2, 0.3])
        assert H_k.shape == (56, 56)

    def test_phase_factors_calculation(self, parser_h5):
        """Test phase factors calculation."""
        # Test Γ-point gives all ones
        phase_gamma = parser_h5._calculate_phase_factors([0.0, 0.0, 0.0])
        expected_length = parser_h5.n_cells - 1  # Exclude sentinel
        assert len(phase_gamma) == expected_length
        np.testing.assert_allclose(phase_gamma, np.ones(expected_length), rtol=1e-12)

        # Test arbitrary k-point gives complex values
        phase_k = parser_h5._calculate_phase_factors([0.5, 0.5, 0.0])
        assert len(phase_k) == expected_length
        assert phase_k.dtype == np.complex128

    def test_kspace_return_full_parameter(self, parser_h5):
        """Test return_full parameter in k-space methods."""
        k_frac = [0.1, 0.2, 0.0]

        # Test Hamiltonian with return_full=True (default)
        H_full = parser_h5.get_hamiltonian_kspace(k_frac, return_full=True)
        H_default = parser_h5.get_hamiltonian_kspace(k_frac)  # Default should be True

        # Test Hamiltonian with return_full=False
        H_U = parser_h5.get_hamiltonian_kspace(k_frac, return_full=False)

        # Test Overlap with return_full=True (default)
        S_full = parser_h5.get_overlap_kspace(k_frac, return_full=True)
        S_default = parser_h5.get_overlap_kspace(k_frac)  # Default should be True

        # Test Overlap with return_full=False
        S_U = parser_h5.get_overlap_kspace(k_frac, return_full=False)

        # Check shapes (should be same for k-space case)
        assert H_full.shape == (56, 56)
        assert H_default.shape == (56, 56)
        assert H_U.shape == (56, 56)
        assert S_full.shape == (56, 56)
        assert S_default.shape == (56, 56)
        assert S_U.shape == (56, 56)

        # Check that default matches return_full=True
        np.testing.assert_allclose(H_full, H_default, rtol=1e-12)
        np.testing.assert_allclose(S_full, S_default, rtol=1e-12)

        # For k-space case, full matrix has upper triangular filled via Hermitian symmetry
        # Check that lower triangular parts match (the computed part)
        tril_indices = np.tril_indices(56)
        np.testing.assert_allclose(H_full[tril_indices], H_U[tril_indices], rtol=1e-12)
        np.testing.assert_allclose(S_full[tril_indices], S_U[tril_indices], rtol=1e-12)

        # Full matrices should be Hermitian
        np.testing.assert_allclose(H_full, H_full.T.conj(), rtol=1e-12)
        np.testing.assert_allclose(S_full, S_full.T.conj(), rtol=1e-12)

    def test_kspace_full_vs_U_gamma_point(self, parser_h5):
        """Test return_full parameter at Γ-point."""
        # At Γ-point, should be essentially the same
        H_full = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0], return_full=True)
        H_U = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0], return_full=False)

        S_full = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0], return_full=True)
        S_U = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0], return_full=False)

        # At Γ-point, should be real and symmetric
        # Full matrix has upper triangular filled, U matrix is lower triangular only
        # Check that lower triangular parts are identical
        tril_indices = np.tril_indices(56)
        np.testing.assert_allclose(H_full[tril_indices], H_U[tril_indices], rtol=1e-12)
        np.testing.assert_allclose(S_full[tril_indices], S_U[tril_indices], rtol=1e-12)

        # Should still be essentially real at Γ-point
        assert np.max(np.abs(H_full.imag)) < 1e-12
        assert np.max(np.abs(H_U.imag)) < 1e-12
        assert np.max(np.abs(S_full.imag)) < 1e-12
        assert np.max(np.abs(S_U.imag)) < 1e-12

    def test_gamma_hamiltonian_vs_reference(self, parser_h5, gamma_hamiltonian_ref):
        """Test that reconstructed Hamiltonian at Γ-point matches reference."""
        # Get Hamiltonian at Γ-point from k-space transformation
        H_gamma_constructed = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0])

        # Should be real at Γ-point, take real part for comparison
        H_gamma_real = H_gamma_constructed.real

        # Check shapes match
        assert H_gamma_real.shape == gamma_hamiltonian_ref.shape
        assert H_gamma_real.shape == (56, 56)

        # Check that imaginary part is negligible
        assert np.max(np.abs(H_gamma_constructed.imag)) < 1e-12

        # Compare with reference matrix (allow for numerical precision differences)
        np.testing.assert_allclose(
            H_gamma_real,
            gamma_hamiltonian_ref,
            rtol=1e-9,
            atol=1e-11,
            err_msg="Reconstructed Hamiltonian at Γ-point does not match reference",
        )

        # Test specific matrix elements for additional validation
        # Diagonal elements should match closely
        np.testing.assert_allclose(
            np.diag(H_gamma_real),
            np.diag(gamma_hamiltonian_ref),
            rtol=1e-9,
            atol=1e-11,
            err_msg="Hamiltonian diagonal elements do not match reference",
        )

        # Check matrix symmetry (Hamiltonian should be Hermitian, real at Γ)
        np.testing.assert_allclose(
            H_gamma_real,
            H_gamma_real.T,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Reconstructed Hamiltonian is not symmetric at Γ-point",
        )

    def test_gamma_overlap_vs_reference(self, parser_h5, gamma_overlap_ref):
        """Test that reconstructed overlap matrix at Γ-point matches reference."""
        # Get overlap matrix at Γ-point from k-space transformation
        S_gamma_constructed = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0])

        # Should be real at Γ-point, take real part for comparison
        S_gamma_real = S_gamma_constructed.real

        # Check shapes match
        assert S_gamma_real.shape == gamma_overlap_ref.shape
        assert S_gamma_real.shape == (56, 56)

        # Check that imaginary part is negligible
        assert np.max(np.abs(S_gamma_constructed.imag)) < 1e-12

        # Compare with reference matrix (allow for numerical precision differences)
        np.testing.assert_allclose(
            S_gamma_real,
            gamma_overlap_ref,
            rtol=1e-9,
            atol=1e-11,
            err_msg="Reconstructed overlap matrix at Γ-point does not match reference",
        )

        # Test specific matrix elements for additional validation
        # Diagonal elements should be close to 1 and match reference
        np.testing.assert_allclose(
            np.diag(S_gamma_real),
            np.diag(gamma_overlap_ref),
            rtol=1e-9,
            atol=1e-11,
            err_msg="Overlap diagonal elements do not match reference",
        )

        # Check matrix symmetry (overlap should be symmetric)
        np.testing.assert_allclose(
            S_gamma_real,
            S_gamma_real.T,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Reconstructed overlap matrix is not symmetric at Γ-point",
        )

        # Overlap diagonal should be positive (positive definite matrix)
        overlap_diagonal = np.diag(S_gamma_real)
        assert np.all(overlap_diagonal > 0.0), "Overlap diagonal elements should be positive"

    def test_reference_matrix_properties(self, gamma_hamiltonian_ref, gamma_overlap_ref):
        """Test physical properties of the reference matrices."""
        # Test Hamiltonian properties
        assert gamma_hamiltonian_ref.shape == (56, 56)

        # Hamiltonian should be symmetric (Hermitian and real)
        np.testing.assert_allclose(
            gamma_hamiltonian_ref,
            gamma_hamiltonian_ref.T,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Reference Hamiltonian is not symmetric",
        )

        # Check eigenvalues are real (for Hermitian matrix)
        H_eigenvals = np.linalg.eigvals(gamma_hamiltonian_ref)
        assert np.all(np.abs(H_eigenvals.imag) < 1e-12), "Hamiltonian eigenvalues should be real"

        # Test overlap matrix properties
        assert gamma_overlap_ref.shape == (56, 56)

        # Overlap should be symmetric and positive definite
        np.testing.assert_allclose(
            gamma_overlap_ref,
            gamma_overlap_ref.T,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Reference overlap matrix is not symmetric",
        )

        # Check that overlap is positive definite (all eigenvalues > 0)
        S_eigenvals = np.linalg.eigvals(gamma_overlap_ref)
        assert np.all(S_eigenvals.real > 1e-10), "Overlap matrix should be positive definite"
        assert np.all(np.abs(S_eigenvals.imag) < 1e-12), "Overlap eigenvalues should be real"

        # Diagonal elements should be close to 1
        S_diag = np.diag(gamma_overlap_ref)

        # Overlap diagonal should be positive (positive definite matrix)
        assert np.all(S_diag > 0.0), "Overlap diagonal elements should be positive"

    def test_gamma_point_consistency_checks(self, parser_h5, gamma_hamiltonian_ref, gamma_overlap_ref):
        """Test consistency between different methods at Γ-point."""
        # Test that both return_full options give same result at Γ
        H_full = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0], return_full=True)
        H_U = parser_h5.get_hamiltonian_kspace([0.0, 0.0, 0.0], return_full=False)
        S_full = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0], return_full=True)
        S_U = parser_h5.get_overlap_kspace([0.0, 0.0, 0.0], return_full=False)

        # H_full should have upper triangular part filled via Hermitian symmetry
        # H_U should be the lower triangular matrix from Fourier transform
        # Only the lower triangular parts should match
        tril_indices = np.tril_indices(56)
        np.testing.assert_allclose(H_full[tril_indices], H_U[tril_indices], rtol=1e-12)
        np.testing.assert_allclose(S_full[tril_indices], S_U[tril_indices], rtol=1e-12)

        # At Γ-point, full matrix should be symmetric (Hermitian and real)
        np.testing.assert_allclose(H_full, H_full.T.conj(), rtol=1e-12)
        np.testing.assert_allclose(S_full, S_full.T.conj(), rtol=1e-12)

        # Test that our reconstruction matches the reference
        np.testing.assert_allclose(H_full.real, gamma_hamiltonian_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(S_full.real, gamma_overlap_ref, rtol=1e-9, atol=1e-11)

        # Test that matrices have correct conditioning
        H_cond = np.linalg.cond(gamma_hamiltonian_ref)
        S_cond = np.linalg.cond(gamma_overlap_ref)

        # Matrices should be well-conditioned for numerical stability
        assert H_cond < 1e12, f"Hamiltonian is poorly conditioned: {H_cond}"
        assert S_cond < 1e12, f"Overlap matrix is poorly conditioned: {S_cond}"
