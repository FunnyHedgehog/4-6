#pragma once
#include <mpi.h>
#include <vector>

/*
 * MPIGroup, MPICommunicator, MPI are lightweight wrappers for underlying MPI functions.
 * They provide more high level interface for MPI functionality access.
 */

class MPIGroup
{
public:
	static MPIGroup makeByInclude(const MPIGroup& base, const std::vector<int>& ranks)
	{
		MPIGroup g;
		// const_cast is safe here, since function below only reads data
		MPI_Group_incl(base.group, ranks.size(), const_cast<int*>(ranks.data()), &g.group);
		return g;
	}

	bool isNull() const
	{
		return group == MPI_GROUP_NULL;
	}

	MPIGroup()
		: group(MPI_GROUP_NULL)
	{}

	MPIGroup(const MPIGroup& other)
	{
		MPI_Group_union(other.group, MPI_GROUP_EMPTY, &group);
	}

	MPIGroup& operator=(const MPIGroup& other)
	{
		if (other.group != group)
			MPI_Group_union(other.group, MPI_GROUP_EMPTY, &group);
		return *this;
	}

	MPIGroup(MPIGroup&& other)
	{
		group = other.group;
		other.group = MPI_GROUP_NULL;
	}

	~MPIGroup()
	{
		if (!isNull())
			MPI_Group_free(&group);
	}

private:
	MPI_Group group;
	friend class MPICommunicator;
};

class MPICommunicator
{
public:
	MPICommunicator split(int color, int rank)
	{
		MPICommunicator c;
		MPI_Comm_split(comm, color, rank, &c.comm);
		return c;
	}

	MPIGroup getGroup() const
	{
		MPIGroup g;
		MPI_Comm_group(comm, &g.group);
		return g;
	}

	int getRank() const
	{
		int rank = -1;
		if (!isNull())
			MPI_Comm_rank(comm, &rank);
		return rank;
	}

	int getSize() const
	{
		int size = -1;
		if (!isNull())
			MPI_Comm_size(comm, &size);
		return size;
	}

	bool isNull() const
	{
		return comm == MPI_COMM_NULL;
	}

	void send(int data, int destinationRank, int messageTag = 0) const
	{
		MPI_Send(&data, 1, MPI_INT, destinationRank, messageTag, comm);
	}

	void sendArray(std::vector<int> data, int destinationRank, int messageTag = 0) const
	{
		MPI_Send(data.data(), data.size(), MPI_INT, destinationRank, messageTag, comm);
	}

	int receive(int sourceRank, int messageTag = MPI_ANY_TAG) const
	{
		int data;
		MPI_Recv(&data, 1, MPI_INT, sourceRank, messageTag, comm, MPI_STATUS_IGNORE);
		return data;
	}

	std::vector<int> receiveArray(int sourceRank, int messageTag = MPI_ANY_TAG) const
	{
		MPI_Status status;
		MPI_Probe(sourceRank, messageTag, comm, &status);
		int sizeProbed;
		MPI_Get_count(&status, MPI_INT, &sizeProbed);
		std::vector<int> data(sizeProbed);
		MPI_Recv(data.data(), sizeProbed, MPI_INT, sourceRank, messageTag, comm, MPI_STATUS_IGNORE);
		return data;
	}

	static MPICommunicator makeBasedOnGroup(const MPIGroup& group, const MPICommunicator& parent)
	{
		MPICommunicator c;
		MPI_Comm_create(parent.comm, group.group, &c.comm);
		return c;
	}

	static void makeInterCommunicator(
		const MPICommunicator& localComm,  // Local(intra)communicator
		int localLeaderRank,               // Rank in local_comm of leader
		const MPICommunicator& peerComm,   // Communicator used to communicate between a designated process in the other communicator.Significant only at the process in local_comm with rank local_leader.
		int remoteLeaderRank,              // Rank in peer_comm of remote leader
		int uniqueTag,                     // Message tag to use in constructing intercommunicator; if multiple MPI_Intercomm_creates are being made, they should use different tags
		MPICommunicator& outInterCommunicator)
	{
		// Two local communicators, which in this context are known as the local leader, connected to the remote leader;
		// The peer communicator, often MPI_COMM_WORLD, that contains the local communicators;
		// The sender specifies as target the local number of the other leader in the other sub-communicator;
		// Likewise, the receiver specifies as source the local number of the sender in its sub-communicator.
		// In one way, this design makes sense : processors are referred to in their natural, local, numbering.On the other hand, 
		// it means that each group needs to know how the local ordering of the other group is arranged.
		MPI_Intercomm_create(localComm.comm, localLeaderRank, peerComm.comm, remoteLeaderRank, uniqueTag, &outInterCommunicator.comm);
	}

	MPICommunicator()
		: comm(MPI_COMM_NULL)
	{}

	MPICommunicator(const MPICommunicator& other)
	{
		MPI_Comm_dup(other.comm, &comm);
	}

	MPICommunicator& operator=(const MPICommunicator& other)
	{
		if (other.comm != comm)
			MPI_Comm_dup(other.comm, &comm);
		return *this;
	}

	MPICommunicator(MPICommunicator&& other)
	{
		comm = other.comm;
		other.comm = MPI_COMM_NULL;
	}

	~MPICommunicator()
	{
		if (!isNull())
			MPI_Comm_free(&comm);
	}

private:
	MPI_Comm comm;
	friend class MPI;
};

class MPI
{
public:
	static MPI& getInst()
	{
		static MPI instance;
		return instance;
	}

	MPICommunicator getWorldComm()
	{
		MPICommunicator c;
		MPI_Comm_dup(MPI_COMM_WORLD, &c.comm);
		return c;
	}

private:
	MPI()
	{
		MPI_Init(nullptr, nullptr);
	}

	~MPI()
	{
		MPI_Finalize();
	}

	MPI(const MPI&) = delete;
	MPI& operator=(const MPI&) = delete;
};