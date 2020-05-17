	#include <iostream>
	#include <algorithm>
	#include <iterator>
	#include "MPILib.h"

	constexpr auto SIZE = 6;

	using Matrix = std::vector<std::vector<int> >;
	enum RollDirection {
		LEFT,
		UP
	};

	void printMatrix(const Matrix & matrix)
	{
		std::cout << std::endl;
		for (auto i : matrix) {
			for (auto j : i) {
				std::cout << j << "\t";
			}
			std::cout << std::endl;
		}
	}
	template <RollDirection direction>
	void Roll(Matrix matrix, int coord, int roll_for) {
		int x, y, last;
		if (direction == LEFT) {
			y = coord;
			x = 0;
			last = matrix[0].size() - 1;
		}
		else {
			x = coord;
			y = 0;
			last = matrix.size() - 1;
		}

		for (int i = 0; i < roll_for; ++i) {
			int first = matrix[y][x];
			for (int j = 0; j < last; ++j) {
				if (direction == LEFT) {
					matrix[y][j] = matrix[y][j + 1];
				}
				else {
					matrix[j][x] = matrix[j + 1][x];
				}
			}
			if (direction == LEFT) {
				matrix[y][last] = first;
			}
			else {
				matrix[last][x] = first;
			}
		}
	}

	Matrix Cannon(Matrix first, const Matrix second)
	{
		Matrix result(first.size(), std::vector<int>(second.size()));

		for (int i = 0; i < first.size(); i++)
			Roll<LEFT>(first, i, i);
		for (int i = 0; i < first.size(); i++)
			Roll<UP>(second, i, i);

	#pragma omp for
		for (int k = 0; k < first.size(); k++)
		{
			for (int i = 0; i < first.size(); i++)
				for (int j = 0; j < first.size(); j++)
				{
					auto m = (i + j + k) % first.size();
	#pragma omp atomic
					result[i][j] += first[i][m] * second[m][j];
					Roll<LEFT>(first, i, 1);
					Roll<UP>(second, j, 1);
				}
		}
		return result;
	}


	void fillMatrix(Matrix &matrixB, std::vector<int> &B);

	void multiplyMatrix(MPICommunicator &interComm, std::vector<int> &resultArray);

	int main(int argc, char* argv[]) {

		auto worldCommunicator = MPI::getInst().getWorldComm();

		if (worldCommunicator.getSize() != 9)
		{
			std::cout << "Number of processes must be 9! Exiting..";
			return 0;
		}

		constexpr int SERVER_RANK = 0;
		int worldRank = worldCommunicator.getRank();
		bool isServer = (worldRank == SERVER_RANK);

		// Split all processes into two subgroups: server and the rest
		auto localComm = worldCommunicator.split(isServer, worldRank);

		constexpr int TAG_FOR_1ST_INTERCOMM = 0;
		constexpr int TAG_FOR_2ND_INTERCOMM = 1;
		if (!isServer)
		{
			auto localGroup = localComm.getGroup();
			auto group1Comm = MPICommunicator::makeBasedOnGroup(MPIGroup::makeByInclude(localGroup, { 0, 1, 2, 3 }), localComm);
			auto group2Comm = MPICommunicator::makeBasedOnGroup(MPIGroup::makeByInclude(localGroup, { 4, 5, 6, 7 }), localComm);

			// One intercommunicator for each group to communicate with server
			MPICommunicator interComm;
			if (!group1Comm.isNull())
			{
				int localLeaderRank = 0;
				MPICommunicator::makeInterCommunicator(group1Comm, localLeaderRank, worldCommunicator, SERVER_RANK, TAG_FOR_1ST_INTERCOMM, interComm);
				if (group1Comm.getRank() == localLeaderRank)
				{
					std::vector<int> resultArray;

					multiplyMatrix(interComm, resultArray);
					interComm.sendArray(resultArray, 0);
				}
			}
			if (!group2Comm.isNull())
			{
				int localLeaderRank = 0;
				MPICommunicator::makeInterCommunicator(group2Comm, localLeaderRank, worldCommunicator, SERVER_RANK, TAG_FOR_2ND_INTERCOMM, interComm);
				if (group2Comm.getRank() == localLeaderRank)
				{
					std::vector<int> resultArray;

					multiplyMatrix(interComm, resultArray);
					interComm.sendArray(resultArray, 0);
				}
			}
		}
		else
		{
			int localLeaderRank = 0;
			int remoteLeaderRank;
			MPICommunicator interCommWith1stGroup;
			MPICommunicator::makeInterCommunicator(localComm, localLeaderRank, worldCommunicator, (remoteLeaderRank = 1), TAG_FOR_1ST_INTERCOMM, interCommWith1stGroup);
			MPICommunicator interCommWith2ndGroup;
			MPICommunicator::makeInterCommunicator(localComm, localLeaderRank, worldCommunicator, (remoteLeaderRank = 5), TAG_FOR_2ND_INTERCOMM, interCommWith2ndGroup);

			std::vector<int> A(SIZE * SIZE);
			std::generate(A.begin(), A.end(), []() { return rand() % 100; });
			std::vector<int> B(SIZE * SIZE);
			std::generate(B.begin(), B.end(), []() { return rand() % 100; });

			std::cout << "Local leader of 1st group is sending data to group1 by intercommunicator...\n";
			interCommWith1stGroup.sendArray(A, 0);
			interCommWith1stGroup.sendArray(B, 0);

			interCommWith2ndGroup.sendArray(A, 0);
			interCommWith2ndGroup.sendArray(B, 0);

			std::cout << "Server has received array with size from 1st group\n";
			std::vector<int> recieved1stArray = interCommWith1stGroup.receiveArray(0);
			std::vector<std::vector<int>> matrixFrom1stGroup(SIZE);
			fillMatrix(matrixFrom1stGroup, recieved1stArray);
			std::cout << std::endl << "Result matrix from 1st group: " << std::endl;
			printMatrix(matrixFrom1stGroup);

			std::cout << "Server has received array with size from 2st group\n";
			std::vector<int> recieved2ndArray = interCommWith2ndGroup.receiveArray(0);
			std::vector<std::vector<int>> matrixFrom2ndGroup(SIZE);
			fillMatrix(matrixFrom2ndGroup, recieved2ndArray);
			std::cout << std::endl << "Result matrix from 2nd group: " << std::endl;
			printMatrix(matrixFrom2ndGroup);

		}

		return 0;
	}

	void multiplyMatrix(MPICommunicator &interComm, std::vector<int> &resultArray)
	{
		std::vector<std::vector<int>> matrixA(SIZE);
		std::vector<std::vector<int>> matrixB(SIZE);

		std::vector<int> A = interComm.receiveArray(0);
		std::vector<int> B = interComm.receiveArray(0);
		std::cout << "Local leader of 1st group is recieving data from server by intercommunicator...\n";
		std::cout << "Recieved array A with size " << A.size() << std::endl;
		std::cout << "Recieved array B with size " << B.size() << std::endl;

		fillMatrix(matrixA, A);
		fillMatrix(matrixB, B);

		printMatrix(matrixA);
		std::cout << std::endl;
		printMatrix(matrixB);

		Matrix result = Cannon(matrixA, matrixB);


		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++) {
				resultArray.push_back(result[i][j]);
			}
		}
	}

	void fillMatrix(Matrix &matrixB, std::vector<int> &B)
	{
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++) {
				matrixB[i].push_back(B[i + j]);
			}
		}
	}
