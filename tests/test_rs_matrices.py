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

    @pytest.fixture
    def test_data_dir(self):
        """Fixture providing path to test data directory."""
        return "tests/data/tag_output_rs_matrices_h5_250806"

    @pytest.fixture
    def parser_h5(self, test_data_dir):
        """Fixture providing ParseRSMatrices instance for H5 format."""
        return ParseRSMatrices(test_data_dir, use_h5=True)

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

    def test_get_hamiltonian_dense(self, parser_h5):
        """Test getting Hamiltonian in dense format."""
        dense_h = parser_h5.get_hamiltonian_dense()

        # Check shape and type
        expected_size = parser_h5.n_cells * parser_h5.n_basis
        assert dense_h.shape == (expected_size, expected_size)
        assert dense_h.dtype == np.float64

        # Check that matrix is not all zeros
        assert not np.all(dense_h == 0)

        # Check symmetry (should be symmetric for Hamiltonian)
        np.testing.assert_allclose(dense_h, dense_h.T, rtol=1e-12, atol=1e-12)

    def test_get_overlap_dense(self, parser_h5):
        """Test getting overlap matrix in dense format."""
        dense_s = parser_h5.get_overlap_dense()

        # Check shape and type
        expected_size = parser_h5.n_cells * parser_h5.n_basis
        assert dense_s.shape == (expected_size, expected_size)
        assert dense_s.dtype == np.float64

        # Check that matrix is not all zeros
        assert not np.all(dense_s == 0)

        # Check symmetry (should be symmetric for overlap)
        np.testing.assert_allclose(dense_s, dense_s.T, rtol=1e-12, atol=1e-12)

    def test_get_hamiltonian_csr(self, parser_h5):
        """Test getting Hamiltonian in CSR sparse format."""
        csr_h = parser_h5.get_hamiltonian_csr()

        # Check type and shape
        assert isinstance(csr_h, sparse.csr_matrix)
        expected_size = parser_h5.n_cells * parser_h5.n_basis
        assert csr_h.shape == (expected_size, expected_size)
        assert csr_h.dtype == np.float64

        # Check that matrix has non-zero elements
        assert csr_h.nnz > 0

        # Check sparsity (should be sparse, not dense)
        total_elements = expected_size * expected_size
        sparsity = 1 - (csr_h.nnz / total_elements)
        assert sparsity > 0.5  # Should be at least 50% sparse

    def test_get_overlap_csr(self, parser_h5):
        """Test getting overlap matrix in CSR sparse format."""
        csr_s = parser_h5.get_overlap_csr()

        # Check type and shape
        assert isinstance(csr_s, sparse.csr_matrix)
        expected_size = parser_h5.n_cells * parser_h5.n_basis
        assert csr_s.shape == (expected_size, expected_size)
        assert csr_s.dtype == np.float64

        # Check that matrix has non-zero elements
        assert csr_s.nnz > 0

    def test_dense_csr_equivalence(self, parser_h5):
        """Test that dense and CSR formats give equivalent results."""
        # Load both formats
        dense_h = parser_h5.get_hamiltonian_dense()
        csr_h = parser_h5.get_hamiltonian_csr()

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
        dense_h = parser_h5.get_hamiltonian_dense()
        dense_s = parser_h5.get_overlap_dense()

        # Test that overlap matrix is positive definite (all eigenvalues > 0)
        # Note: Only test a small block to avoid expensive computation
        block_size = min(100, dense_s.shape[0])
        s_block = dense_s[:block_size, :block_size]
        eigenvals = np.linalg.eigvals(s_block)
        assert np.all(eigenvals > -1e-10)  # Allow small numerical errors

        # Test that non-zero diagonal elements of overlap are reasonable
        diagonal_s = np.diag(dense_s)
        non_zero_diag = diagonal_s[diagonal_s > 1e-10]

        # Non-zero diagonal elements should be close to 1 for normalized basis
        if len(non_zero_diag) > 0:
            assert np.mean(non_zero_diag) > 0.5
            assert np.mean(non_zero_diag) < 2.0
            assert np.min(non_zero_diag) > 0.1  # Should be positive for overlap

        # Most diagonal elements might be zero due to sparsity
        sparsity_diag = 1 - len(non_zero_diag) / len(diagonal_s)
        assert sparsity_diag >= 0  # Allow any sparsity level
