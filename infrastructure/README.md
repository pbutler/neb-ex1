# Cluster creation instructions

1. Clone/download [soperator terraform](https://github.com/nebius/nebius-solutions-library/tree/main/soperator) setup scripts and follow the instructions in the README.md file.
  a. After step 2, copy the [terraform.tfvars](./terraform.tfvars) file to the
    `terraform.tfvars` file in the installations directory.
2. You'll still need to set the following values:
      * `company_name` -- your company name, used for the cluster name.
      * `filestore_jail.existing.id`  -- this is the ID of the existing filestore
       instance that you want to use for the jail.
      * `filestore_jail_submounts[0]..existing.id` -- this is the iD of the existing
        submount that you want to use for the jail data.
      * `slurm_operator_version` -- the version of the slurm operator you want
        to use
      * `slurm_login_ssh_root_public_keys[0]` - the public SSH key that you
        want to use for the root user on the login node.
